//! Stereo-seq GEM: the text form of a spatial expression matrix.
//!
//! A GEM is a tab-separated table with a `#`-prefixed header carrying the chip id, bin size and
//! coordinate offsets. Whole-embryo runs produce tens of gigabytes of it, and Python's csv/pandas
//! path is where people actually run out of memory — so this parses in Rust, chunked by bytes,
//! and hands Arrow batches across without a copy.
//!
//! Header lines look like:
//! ```text
//! #FileFormat=GEMv0.1
//! #SortedBy=None
//! #BinSize=1
//! #StereoChip=SS200000135TL_D1
//! #OffsetX=0
//! #OffsetY=0
//! geneID    x    y    MIDCount
//! ```
//! Column names vary between SAW versions (`geneID`/`geneName`, `MIDCount`/`MIDCounts`/`UMICount`,
//! optional `ExonCount`), so columns are located by name from the header rather than by position.

use crate::error::{GemingaError, Result};
use arrow::array::{
    Array, ArrayRef, Int32Builder, RecordBatch, StringDictionaryBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use flate2::read::MultiGzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct GemHeader {
    pub fields: HashMap<String, String>,
    pub columns: Vec<String>,
    /// Byte offset of the first data row.
    pub data_offset: u64,
}

impl GemHeader {
    pub fn bin_size(&self) -> u32 {
        self.fields
            .get("BinSize")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1)
    }
    pub fn offset_x(&self) -> i32 {
        self.fields
            .get("OffsetX")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    }
    pub fn offset_y(&self) -> i32 {
        self.fields
            .get("OffsetY")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    }
    pub fn chip(&self) -> Option<&str> {
        self.fields
            .get("StereoChip")
            .map(|s| s.as_str())
            .or_else(|| self.fields.get("Stereo-seqChip").map(|s| s.as_str()))
    }
}

fn open_reader(path: &str) -> Result<Box<dyn BufRead + Send>> {
    let f = File::open(path)?;
    if path.ends_with(".gz") || path.ends_with(".bgz") {
        Ok(Box::new(BufReader::with_capacity(
            1 << 20,
            MultiGzDecoder::new(f),
        )))
    } else {
        Ok(Box::new(BufReader::with_capacity(1 << 20, f)))
    }
}

/// Read only the `#` header and the column line. Cheap: a few hundred bytes.
pub fn read_header(path: &str) -> Result<GemHeader> {
    let mut r = open_reader(path)?;
    let mut h = GemHeader::default();
    let mut line = String::new();
    let mut consumed = 0u64;
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        consumed += n as u64;
        let t = line.trim_end_matches(['\n', '\r']);
        if let Some(rest) = t.strip_prefix('#') {
            if let Some((k, v)) = rest.split_once('=') {
                h.fields.insert(k.trim().to_string(), v.trim().to_string());
            }
        } else {
            h.columns = t.split('\t').map(|s| s.trim().to_string()).collect();
            h.data_offset = consumed;
            break;
        }
    }
    if h.columns.is_empty() {
        return Err(GemingaError::Invalid(format!(
            "{path}: no column header row found; is this a GEM file?"
        )));
    }
    Ok(h)
}

/// Locate a column by any of several accepted names.
fn find(cols: &[String], names: &[&str]) -> Option<usize> {
    for n in names {
        if let Some(i) = cols.iter().position(|c| c.eq_ignore_ascii_case(n)) {
            return Some(i);
        }
    }
    None
}

pub fn gem_schema(with_exon: bool) -> SchemaRef {
    let mut f = vec![
        Field::new(
            "gene",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
        Field::new("count", DataType::UInt32, false),
    ];
    if with_exon {
        f.push(Field::new("exon", DataType::UInt32, true));
    }
    Arc::new(Schema::new(f))
}

pub struct GemChunker {
    reader: Box<dyn BufRead + Send>,
    schema: SchemaRef,
    i_gene: usize,
    i_x: usize,
    i_y: usize,
    i_count: usize,
    i_exon: Option<usize>,
    target_bytes: usize,
    /// Coordinates are stored relative to the chip origin; add the header offsets back.
    off_x: i32,
    off_y: i32,
    pub header: GemHeader,
    pub rows_read: u64,
    pub bytes_read: u64,
    exhausted: bool,
}

impl GemChunker {
    /// The chunk this reader plans to materialize next (its lease is taken before reading).
    pub fn target_bytes(&self) -> usize {
        self.target_bytes
    }

    pub fn new(path: &str, target_bytes: usize, apply_offset: bool) -> Result<Self> {
        let header = read_header(path)?;
        let cols = &header.columns;
        let i_gene =
            find(cols, &["geneID", "geneName", "gene", "MIDCounts_geneID"]).ok_or_else(|| {
                GemingaError::Invalid(format!("GEM header has no gene column; saw {cols:?}"))
            })?;
        let i_x = find(cols, &["x"]).ok_or_else(|| {
            GemingaError::Invalid(format!("GEM header has no x column; saw {cols:?}"))
        })?;
        let i_y = find(cols, &["y"]).ok_or_else(|| {
            GemingaError::Invalid(format!("GEM header has no y column; saw {cols:?}"))
        })?;
        let i_count = find(
            cols,
            &["MIDCount", "MIDCounts", "UMICount", "count", "values"],
        )
        .ok_or_else(|| {
            GemingaError::Invalid(format!("GEM header has no count column; saw {cols:?}"))
        })?;
        let i_exon = find(cols, &["ExonCount", "exon"]);

        // Skip past the header we already consumed.
        let mut reader = open_reader(path)?;
        let mut skip = String::new();
        loop {
            skip.clear();
            let n = reader.read_line(&mut skip)?;
            if n == 0 || !skip.starts_with('#') {
                break; // the last line read is the column header itself
            }
        }

        Ok(Self {
            reader,
            schema: gem_schema(i_exon.is_some()),
            i_gene,
            i_x,
            i_y,
            i_count,
            i_exon,
            target_bytes: target_bytes.max(4096),
            off_x: if apply_offset { header.offset_x() } else { 0 },
            off_y: if apply_offset { header.offset_y() } else { 0 },
            header,
            rows_read: 0,
            bytes_read: 0,
            exhausted: false,
        })
    }

    #[allow(dead_code)]
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Next chunk of rows, sized by bytes consumed rather than row count.
    /// Optionally keep only rows inside a spatial window — the cheapest possible pushdown,
    /// applied before anything enters an Arrow buffer.
    pub fn next_chunk(
        &mut self,
        window: Option<(i32, i32, i32, i32)>,
    ) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        let mut genes: StringDictionaryBuilder<Int32Type> = StringDictionaryBuilder::new();
        let mut xs = Int32Builder::new();
        let mut ys = Int32Builder::new();
        let mut cs = UInt32Builder::new();
        let mut es = UInt32Builder::new();
        let mut consumed = 0usize;
        let mut kept = 0usize;
        let mut line = String::new();
        let need = self.i_gene.max(self.i_x).max(self.i_y).max(self.i_count);

        while consumed < self.target_bytes {
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                self.exhausted = true;
                break;
            }
            consumed += n;
            self.bytes_read += n as u64;
            let t = line.trim_end_matches(['\n', '\r']);
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = t.split('\t').collect();
            if f.len() <= need {
                continue; // ragged line; skip rather than abort a 40 GB parse
            }
            let x: i32 = match f[self.i_x].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let y: i32 = match f[self.i_y].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let x = x + self.off_x;
            let y = y + self.off_y;
            if let Some((x0, x1, y0, y1)) = window {
                if x < x0 || x >= x1 || y < y0 || y >= y1 {
                    self.rows_read += 1;
                    continue;
                }
            }
            let c: u32 = match f[self.i_count].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            genes.append_value(f[self.i_gene]);
            xs.append_value(x);
            ys.append_value(y);
            cs.append_value(c);
            if let Some(ie) = self.i_exon {
                match f.get(ie).and_then(|s| s.parse::<u32>().ok()) {
                    Some(v) => es.append_value(v),
                    None => es.append_null(),
                }
            }
            self.rows_read += 1;
            kept += 1;
        }

        if kept == 0 {
            return Ok(if self.exhausted {
                None
            } else {
                Some(RecordBatch::new_empty(self.schema.clone()))
            });
        }
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(genes.finish()),
            Arc::new(xs.finish()),
            Arc::new(ys.finish()),
            Arc::new(cs.finish()),
        ];
        if self.i_exon.is_some() {
            cols.push(Arc::new(es.finish()));
        }
        Ok(Some(RecordBatch::try_new(self.schema.clone(), cols)?))
    }
}

/// Scan a GEM once to learn its spatial extent, row count and gene count — everything a view
/// needs to plan, without holding the file. Returns (minx, maxx, miny, maxy, rows, genes).
pub fn gem_extent(path: &str) -> Result<(i32, i32, i32, i32, u64, u64)> {
    let mut ch = GemChunker::new(path, 8 << 20, true)?;
    let (mut x0, mut x1, mut y0, mut y1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    let mut rows = 0u64;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    while let Some(b) = ch.next_chunk(None)? {
        if b.num_rows() == 0 {
            continue;
        }
        let xs = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let ys = b
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        for i in 0..b.num_rows() {
            let (x, y) = (xs.value(i), ys.value(i));
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        let g = arrow::array::cast::as_dictionary_array::<Int32Type>(b.column(0));
        let vals = g
            .values()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        for i in 0..vals.len() {
            seen.insert(vals.value(i).to_string());
        }
        rows += b.num_rows() as u64;
    }
    if rows == 0 {
        return Err(GemingaError::Invalid(format!("{path}: no data rows")));
    }
    Ok((x0, x1 + 1, y0, y1 + 1, rows, seen.len() as u64))
}

// ---------------------------------------------------------------------------
// Spatial level-of-detail index
// ---------------------------------------------------------------------------

/// One pass over a GEM yields everything needed to navigate it afterwards without touching
/// the file again: chip extent, a spatial density pyramid, per-gene totals, and the occupancy
/// grid the cost model runs on.
/// Bumped when the index's meaning changes; an older sidecar fails to load and is rebuilt.
pub const GEM_INDEX_VERSION: u32 = 2;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GemIndex {
    #[serde(default)]
    pub version: u32,
    pub path: String,
    pub file_bytes: u64,
    pub n_rows: u64,
    pub total_mid: u64,
    pub bin_size: u32,
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
    pub grid: usize,
    /// `grid x grid` MID density over the chip extent, row-major (y outer, x inner).
    pub density: Vec<f32>,
    /// Observations per grid cell — what the cost estimate is derived from.
    pub occupancy: Vec<u32>,
    pub genes: Vec<String>,
    pub gene_totals: Vec<u64>,
    pub bytes_per_row: f64,
    pub has_exon: bool,
    pub chip: Option<String>,
}

impl GemIndex {
    pub fn build(path: &str, grid: usize) -> Result<Self> {
        use arrow::array::{DictionaryArray, Int32Array, StringArray, UInt32Array};
        use arrow::datatypes::Int32Type;
        use std::collections::HashMap;

        let grid = grid.clamp(16, 4096);
        let header = read_header(path)?;
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let (min_x, max_x, min_y, max_y, _rows_hint, _genes_hint) = gem_extent(path)?;
        // gem_extent returns half-open bounds (max + 1), so the width is the plain difference.
        let w = (max_x - min_x).max(1) as f64;
        let h = (max_y - min_y).max(1) as f64;

        let mut density = vec![0f32; grid * grid];
        let mut occupancy = vec![0u32; grid * grid];
        let mut gene_ids: HashMap<String, usize> = HashMap::new();
        let mut gene_totals: Vec<u64> = Vec::new();
        let mut n_rows = 0u64;
        let mut total_mid = 0u64;

        // Offsets applied, as `gem_extent` and `GemStream` do: a file with #OffsetX/#OffsetY
        // must index the same coordinates it streams, or every window lands in cell (0,0).
        let mut ch = GemChunker::new(path, 32 << 20, true)?;
        while let Some(b) = ch.next_chunk(None)? {
            let gi = b.schema().index_of("gene").unwrap_or(0);
            let xi = b.schema().index_of("x").unwrap_or(1);
            let yi = b.schema().index_of("y").unwrap_or(2);
            let ci = b.schema().index_of("count").unwrap_or(3);
            let g = b
                .column(gi)
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| GemingaError::Msg("gene column is not dictionary-encoded".into()))?;
            let vals = g
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GemingaError::Msg("gene dictionary is not utf8".into()))?;
            let xs = b.column(xi).as_any().downcast_ref::<UInt32Array>();
            let xi32 = b.column(xi).as_any().downcast_ref::<Int32Array>();
            let ys = b.column(yi).as_any().downcast_ref::<UInt32Array>();
            let yi32 = b.column(yi).as_any().downcast_ref::<Int32Array>();
            let cs_i32 = b.column(ci).as_any().downcast_ref::<Int32Array>();
            let cs_u32 = b.column(ci).as_any().downcast_ref::<UInt32Array>();
            let cs_u16 = b
                .column(ci)
                .as_any()
                .downcast_ref::<arrow::array::UInt16Array>();

            for i in 0..b.num_rows() {
                let xv = match (xs, xi32) {
                    (Some(a), _) => a.value(i) as i64,
                    (None, Some(a)) => a.value(i) as i64,
                    _ => continue,
                };
                let yv = match (ys, yi32) {
                    (Some(a), _) => a.value(i) as i64,
                    (None, Some(a)) => a.value(i) as i64,
                    _ => continue,
                };
                let cv = match (cs_i32, cs_u32, cs_u16) {
                    (Some(a), _, _) => a.value(i).max(0) as u64,
                    (None, Some(a), _) => a.value(i) as u64,
                    (None, None, Some(a)) => a.value(i) as u64,
                    _ => 1,
                };
                let gx = (((xv - min_x as i64) as f64 / w) * grid as f64) as usize;
                let gy = (((yv - min_y as i64) as f64 / h) * grid as f64) as usize;
                let idx = gy.min(grid - 1) * grid + gx.min(grid - 1);
                density[idx] += cv as f32;
                occupancy[idx] += 1;

                let key = vals.value(g.keys().value(i) as usize);
                let id = match gene_ids.get(key) {
                    Some(&id) => id,
                    None => {
                        let id = gene_totals.len();
                        gene_ids.insert(key.to_string(), id);
                        gene_totals.push(0);
                        id
                    }
                };
                gene_totals[id] += cv;
                total_mid += cv;
                n_rows += 1;
            }
        }
        if n_rows == 0 {
            return Err(GemingaError::Invalid(format!("{path}: no data rows")));
        }
        let mut genes = vec![String::new(); gene_totals.len()];
        for (k, &v) in &gene_ids {
            genes[v] = k.clone();
        }

        Ok(Self {
            version: GEM_INDEX_VERSION,
            path: path.to_string(),
            file_bytes,
            n_rows,
            total_mid,
            bin_size: header.bin_size(),
            min_x,
            max_x,
            min_y,
            max_y,
            grid,
            density,
            occupancy,
            genes,
            gene_totals,
            bytes_per_row: file_bytes as f64 / n_rows as f64,
            has_exon: header
                .columns
                .iter()
                .any(|c| c.to_ascii_lowercase().contains("exon")),
            chip: header.chip().map(|s| s.to_string()),
        })
    }

    fn gx(&self, x: i32) -> usize {
        let w = (self.max_x - self.min_x).max(1) as f64;
        ((((x.max(self.min_x) - self.min_x) as f64 / w) * self.grid as f64) as usize)
            .min(self.grid - 1)
    }
    fn gy(&self, y: i32) -> usize {
        let h = (self.max_y - self.min_y).max(1) as f64;
        ((((y.max(self.min_y) - self.min_y) as f64 / h) * self.grid as f64) as usize)
            .min(self.grid - 1)
    }

    /// Observations inside a spatial window, straight from the occupancy grid. Costs nothing.
    pub fn rows_in(&self, x0: i32, x1: i32, y0: i32, y1: i32) -> u64 {
        // Windows are half-open: the last cell counted is the one holding x1 - 1.
        let (gx0, gx1) = (self.gx(x0), self.gx((x1 - 1).max(x0)));
        let (gy0, gy1) = (self.gy(y0), self.gy((y1 - 1).max(y0)));
        let mut n = 0u64;
        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                n += self.occupancy[gy * self.grid + gx] as u64;
            }
        }
        n
    }

    /// The density grid re-binned over a window — the zoom operation, pure arithmetic.
    pub fn zoom(&self, x0: i32, x1: i32, y0: i32, y1: i32, target: usize) -> (usize, Vec<f32>) {
        let target = target.clamp(4, self.grid);
        let (gx0, gx1) = (self.gx(x0), self.gx((x1 - 1).max(x0)));
        let (gy0, gy1) = (self.gy(y0), self.gy((y1 - 1).max(y0)));
        let (sw, sh) = (gx1 - gx0 + 1, gy1 - gy0 + 1);
        let mut out = vec![0f32; target * target];
        for gy in 0..sh {
            let oy = gy * target / sh;
            for gx in 0..sw {
                let ox = gx * target / sw;
                out[oy * target + ox] += self.density[(gy0 + gy) * self.grid + gx0 + gx];
            }
        }
        (target, out)
    }

    pub fn top_genes(&self, k: usize) -> Vec<(String, u64)> {
        let mut idx: Vec<usize> = (0..self.genes.len()).collect();
        idx.sort_unstable_by_key(|&i| std::cmp::Reverse(self.gene_totals[i]));
        idx.into_iter()
            .take(k)
            .map(|i| (self.genes[i].clone(), self.gene_totals[i]))
            .collect()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        serde_json::to_writer(std::io::BufWriter::new(std::fs::File::create(path)?), self)?;
        Ok(())
    }
    pub fn load(path: &str) -> Result<Self> {
        let idx: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if idx.version != GEM_INDEX_VERSION {
            return Err(GemingaError::Invalid(format!(
                "GEM index {path} was built by an older GEMINGA (version {}, this is {}); rebuild it",
                idx.version, GEM_INDEX_VERSION
            )));
        }
        Ok(idx)
    }
}
