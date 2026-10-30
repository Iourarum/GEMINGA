//! A trigram index over a line-oriented text file — the shape of Microsoft's `tgrep`.
//!
//! The file is cut into blocks of `block_lines` lines. Every block's distinct case-folded byte
//! trigrams go into posting lists (trigram -> sorted block ids, delta-varint encoded). A
//! literal query is turned into its trigrams, their posting lists are intersected, and only the
//! surviving blocks are read and scanned for the literal itself. A term that occurs in a few
//! blocks — a gene, a file name, a selection id — costs a few block reads whatever the size of
//! the file; a term that occurs everywhere degrades to the scan `grep` would do anyway.
//!
//! A JSON log repeats its keys on every line, so a few thousand trigrams occur in every block.
//! Those carry no information and would dominate the index; they are kept as a one-byte
//! "everywhere" mark instead of a list, demoted to a real list the first time a block lacks them.
//!
//! The index covers a prefix of the file ending at a line boundary (`indexed_bytes`). An
//! append-only log is extended with `update()`, which indexes the lines that arrived since;
//! `search()` also scans whatever tail is not indexed yet, so results are complete even when
//! the index lags. The sidecar is a small binary file (`<log>.tgx`), versioned and
//! self-describing; a stale or foreign one is rebuilt.

use crate::error::{GemingaError, Result};
use memchr::{memchr, memchr_iter, memmem, memrchr};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

const MAGIC: &[u8; 4] = b"GMTG";
const VERSION: u32 = 1;

#[derive(Clone)]
enum Posting {
    /// Present in every block indexed so far.
    All,
    /// Sorted block ids, delta-varint encoded; `last` is the final id, to append without decoding.
    List { enc: Vec<u8>, last: u32 },
}

#[derive(Clone)]
pub struct TrigramIndex {
    pub source: String,
    pub block_lines: u32,
    /// Lines already indexed in the last block (it may be partial; `update` fills it first).
    last_block_lines: u32,
    /// The file prefix the index covers; always ends at a line boundary.
    pub indexed_bytes: u64,
    pub n_lines: u64,
    /// Byte offset where block `i` starts.
    blocks: Vec<u64>,
    postings: HashMap<u32, Posting>,
}

#[inline]
fn fold(b: u8) -> u8 {
    b.to_ascii_lowercase()
}

#[inline]
fn key(a: u8, b: u8, c: u8) -> u32 {
    ((fold(a) as u32) << 16) | ((fold(b) as u32) << 8) | fold(c) as u32
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let mut v: u32 = 0;
    let mut shift = 0;
    loop {
        let b = *buf.get(*pos)?;
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
        if shift > 28 {
            return None;
        }
    }
}

fn decode_list(enc: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut cur: u32 = 0;
    let mut first = true;
    while let Some(d) = get_varint(enc, &mut pos) {
        cur = if first { d } else { cur + d };
        first = false;
        out.push(cur);
    }
    out
}

/// `0, 1, ..., n-1` encoded: what an "everywhere" mark becomes when a block lacks the trigram.
fn encode_range(n: u32) -> Vec<u8> {
    let mut enc = Vec::with_capacity(n as usize);
    for i in 0..n {
        put_varint(&mut enc, if i == 0 { 0 } else { 1 });
    }
    enc
}

/// The distinct trigrams of a block's bytes, case-folded. A trigram never spans a line break,
/// since no query contains one.
fn trigrams_of(bytes: &[u8], seen: &mut Vec<u32>) {
    seen.clear();
    if bytes.len() < 3 {
        return;
    }
    for w in bytes.windows(3) {
        if w[0] == b'\n' || w[1] == b'\n' || w[2] == b'\n' {
            continue;
        }
        seen.push(key(w[0], w[1], w[2]));
    }
    seen.sort_unstable();
    seen.dedup();
}

/// Scan one block (`buf`, ending at a line boundary) for the needle: one pass over the bytes,
/// lines materialised only where it hits. `folded` is the block lower-cased when the search
/// ignores case; offsets are the same in both.
fn scan_block(
    buf: &[u8],
    first_line: u64,
    finder: &memmem::Finder<'_>,
    ignore_case: bool,
    folded: &mut Vec<u8>,
    out: &mut Vec<(u64, String)>,
) {
    let hay: &[u8] = if ignore_case {
        folded.clear();
        folded.extend_from_slice(buf);
        folded.make_ascii_lowercase();
        folded
    } else {
        buf
    };
    if finder.needle().is_empty() {
        let mut ln = first_line;
        let mut start = 0usize;
        while start < buf.len() {
            let end = memchr(b'\n', &buf[start..])
                .map(|i| start + i)
                .unwrap_or(buf.len());
            out.push((ln, text_of(&buf[start..end])));
            ln += 1;
            start = end + 1;
        }
        return;
    }
    let mut line_no = first_line;
    let mut counted_to = 0usize;
    let mut reported: Option<usize> = None;
    for pos in finder.find_iter(hay) {
        line_no += memchr_iter(b'\n', &hay[counted_to..pos]).count() as u64;
        counted_to = pos;
        let ls = memrchr(b'\n', &hay[..pos]).map(|i| i + 1).unwrap_or(0);
        if reported == Some(ls) {
            continue; // a second hit on the same line
        }
        reported = Some(ls);
        let le = memchr(b'\n', &hay[pos..])
            .map(|i| pos + i)
            .unwrap_or(hay.len());
        out.push((line_no, text_of(&buf[ls..le])));
    }
}

fn text_of(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw.strip_suffix(b"\r").unwrap_or(raw)).into_owned()
}

impl TrigramIndex {
    /// Index `path` from its first byte. Blocks of `block_lines` lines (64 is a good default:
    /// about 20 KB of journal per block).
    pub fn build(path: &str, block_lines: u32) -> Result<Self> {
        if block_lines == 0 {
            return Err(GemingaError::Invalid("block_lines must be positive".into()));
        }
        let mut idx = TrigramIndex {
            source: path.to_string(),
            block_lines,
            last_block_lines: 0,
            indexed_bytes: 0,
            n_lines: 0,
            blocks: Vec::new(),
            postings: HashMap::new(),
        };
        idx.update()?;
        Ok(idx)
    }

    /// Index the lines appended since the index was built or last updated. Returns how many.
    pub fn update(&mut self) -> Result<u64> {
        let mut f = File::open(&self.source)?;
        let size = f.metadata()?.len();
        if size < self.indexed_bytes {
            return Err(GemingaError::Invalid(format!(
                "{} is shorter ({} bytes) than the {} bytes its index covers: rebuilt, not updated",
                self.source, size, self.indexed_bytes
            )));
        }
        if size == self.indexed_bytes {
            return Ok(0);
        }
        f.seek(SeekFrom::Start(self.indexed_bytes))?;
        let mut rd = BufReader::with_capacity(1 << 20, f);
        let mut line = Vec::with_capacity(512);
        let mut block = Vec::with_capacity(1 << 16);
        let mut in_block = self.last_block_lines;
        let mut extending = in_block > 0; // the last block is partial: fill it first
        let mut block_id: u32 = if extending {
            (self.blocks.len() - 1) as u32
        } else {
            self.blocks.len() as u32
        };
        let mut seen = Vec::with_capacity(4096);
        let mut added: u64 = 0;
        let mut pos = self.indexed_bytes;
        loop {
            line.clear();
            let n = rd.read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if *line.last().unwrap() != b'\n' {
                break; // a torn tail: not a line yet
            }
            if in_block == 0 {
                self.blocks.push(pos); // this line starts a new block
            }
            pos += n as u64;
            block.extend_from_slice(&line);
            in_block += 1;
            self.n_lines += 1;
            added += 1;
            if in_block == self.block_lines {
                self.flush_block(block_id, &block, extending, &mut seen);
                block.clear();
                block_id += 1;
                in_block = 0;
                extending = false;
            }
        }
        if !block.is_empty() {
            self.flush_block(block_id, &block, extending, &mut seen);
        }
        self.last_block_lines = in_block;
        self.indexed_bytes = pos;
        Ok(added)
    }

    /// Add one block's trigrams. `extending` means the block was flushed before with its first
    /// lines, so an "everywhere" trigram missing from these *new* lines is not demoted.
    fn flush_block(&mut self, block_id: u32, bytes: &[u8], extending: bool, seen: &mut Vec<u32>) {
        trigrams_of(bytes, seen);
        if !extending && block_id > 0 {
            // every trigram marked "everywhere" that this block lacks becomes a real list
            let demote: Vec<u32> = self
                .postings
                .iter()
                .filter(|(k, v)| matches!(v, Posting::All) && seen.binary_search(k).is_err())
                .map(|(k, _)| *k)
                .collect();
            for k in demote {
                self.postings.insert(
                    k,
                    Posting::List {
                        enc: encode_range(block_id),
                        last: block_id - 1,
                    },
                );
            }
        }
        for &t in seen.iter() {
            match self.postings.get_mut(&t) {
                Some(Posting::All) => {}
                Some(Posting::List { enc, last }) => {
                    if *last != block_id {
                        put_varint(enc, block_id - *last);
                        *last = block_id;
                    }
                }
                None => {
                    if block_id == 0 {
                        self.postings.insert(t, Posting::All);
                    } else {
                        let mut enc = Vec::new();
                        put_varint(&mut enc, block_id);
                        self.postings.insert(
                            t,
                            Posting::List {
                                enc,
                                last: block_id,
                            },
                        );
                    }
                }
            }
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn n_trigrams(&self) -> usize {
        self.postings.len()
    }

    /// Trigrams present in every block (kept as a mark, not a list).
    pub fn n_everywhere(&self) -> usize {
        self.postings
            .values()
            .filter(|v| matches!(v, Posting::All))
            .count()
    }

    pub fn postings_bytes(&self) -> usize {
        self.postings
            .values()
            .map(|v| match v {
                Posting::All => 1,
                Posting::List { enc, .. } => enc.len(),
            })
            .sum()
    }

    /// Block ids that can hold `pattern` (case-folded). `None` means "every block": the
    /// pattern is shorter than a trigram, or all of its trigrams occur everywhere, so the
    /// index cannot narrow it.
    pub fn candidates(&self, pattern: &[u8]) -> Option<Vec<u32>> {
        if pattern.len() < 3 {
            return None;
        }
        let mut keys: Vec<u32> = pattern.windows(3).map(|w| key(w[0], w[1], w[2])).collect();
        keys.sort_unstable();
        keys.dedup();
        let mut lists: Vec<Vec<u32>> = Vec::new();
        for k in keys {
            match self.postings.get(&k) {
                None => return Some(Vec::new()),
                Some(Posting::All) => {}
                Some(Posting::List { enc, .. }) => lists.push(decode_list(enc)),
            }
        }
        if lists.is_empty() {
            return None;
        }
        lists.sort_by_key(|l| l.len());
        let mut acc = lists[0].clone();
        for l in &lists[1..] {
            let mut out = Vec::with_capacity(acc.len().min(l.len()));
            let (mut i, mut j) = (0, 0);
            while i < acc.len() && j < l.len() {
                if acc[i] == l[j] {
                    out.push(acc[i]);
                    i += 1;
                    j += 1;
                } else if acc[i] < l[j] {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            acc = out;
            if acc.is_empty() {
                break;
            }
        }
        Some(acc)
    }

    /// Lines containing `pattern` as `(line_number, text)`, line numbers from 0. With
    /// `ignore_case` the comparison folds ASCII case on both sides. Lines appended after the
    /// indexed prefix are scanned as well, so the answer is complete for the file as it is.
    pub fn search(&self, pattern: &str, ignore_case: bool) -> Result<Vec<(u64, String)>> {
        let pat: Vec<u8> = if ignore_case {
            pattern.bytes().map(fold).collect()
        } else {
            pattern.as_bytes().to_vec()
        };
        let finder = memmem::Finder::new(&pat);
        let mut out = Vec::new();
        let mut f = File::open(&self.source)?;
        let size = f.metadata()?.len();
        let mut buf = Vec::new();
        let mut folded = Vec::new();
        let ids: Vec<u32> = match self.candidates(&pat) {
            None => (0..self.blocks.len() as u32).collect(), // cannot narrow: every block
            Some(ids) => ids,
        };
        // consecutive candidate blocks are read in one go
        let mut i = 0;
        while i < ids.len() {
            let mut j = i;
            while j + 1 < ids.len() && ids[j + 1] == ids[j] + 1 {
                j += 1;
            }
            let start = self.blocks[ids[i] as usize];
            let end = if (ids[j] as usize) + 1 < self.blocks.len() {
                self.blocks[ids[j] as usize + 1]
            } else {
                self.indexed_bytes
            };
            f.seek(SeekFrom::Start(start))?;
            buf.resize((end - start) as usize, 0);
            f.read_exact(&mut buf)?;
            scan_block(
                &buf,
                ids[i] as u64 * self.block_lines as u64,
                &finder,
                ignore_case,
                &mut folded,
                &mut out,
            );
            i = j + 1;
        }
        if size > self.indexed_bytes {
            // the unindexed tail: complete lines only, scanned the plain way
            f.seek(SeekFrom::Start(self.indexed_bytes))?;
            buf.clear();
            f.read_to_end(&mut buf)?;
            if let Some(last_nl) = buf.iter().rposition(|&b| b == b'\n') {
                buf.truncate(last_nl + 1);
                scan_block(
                    &buf,
                    self.n_lines,
                    &finder,
                    ignore_case,
                    &mut folded,
                    &mut out,
                );
            }
        }
        Ok(out)
    }

    // ---- the sidecar ----
    pub fn save(&self, path: &str) -> Result<()> {
        let mut w = Vec::with_capacity(64 + self.postings_bytes() + 8 * self.blocks.len());
        w.extend_from_slice(MAGIC);
        w.extend_from_slice(&VERSION.to_le_bytes());
        w.extend_from_slice(&self.block_lines.to_le_bytes());
        w.extend_from_slice(&self.last_block_lines.to_le_bytes());
        w.extend_from_slice(&self.indexed_bytes.to_le_bytes());
        w.extend_from_slice(&self.n_lines.to_le_bytes());
        let src = self.source.as_bytes();
        w.extend_from_slice(&(src.len() as u32).to_le_bytes());
        w.extend_from_slice(src);
        w.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for b in &self.blocks {
            w.extend_from_slice(&b.to_le_bytes());
        }
        let mut keys: Vec<&u32> = self.postings.keys().collect();
        keys.sort();
        w.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for k in keys {
            w.extend_from_slice(&k.to_le_bytes());
            match &self.postings[k] {
                Posting::All => w.push(0),
                Posting::List { enc, last } => {
                    w.push(1);
                    w.extend_from_slice(&(enc.len() as u32).to_le_bytes());
                    w.extend_from_slice(enc);
                    w.extend_from_slice(&last.to_le_bytes());
                }
            }
        }
        let tmp = format!("{path}.tmp{}", std::process::id());
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&w)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut buf = Vec::new();
        File::open(path)?.read_to_end(&mut buf)?;
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
            if *p + n > buf.len() {
                return Err(GemingaError::Invalid(format!(
                    "{path}: truncated trigram index"
                )));
            }
            let s = &buf[*p..*p + n];
            *p += n;
            Ok(s)
        };
        if take(&mut p, 4)? != MAGIC {
            return Err(GemingaError::Invalid(format!(
                "{path}: not a GEMINGA trigram index"
            )));
        }
        let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
        let u64_at = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let version = u32_at(take(&mut p, 4)?);
        if version != VERSION {
            return Err(GemingaError::Invalid(format!(
                "{path}: trigram index version {version}, this core writes {VERSION}"
            )));
        }
        let block_lines = u32_at(take(&mut p, 4)?);
        let last_block_lines = u32_at(take(&mut p, 4)?);
        let indexed_bytes = u64_at(take(&mut p, 8)?);
        let n_lines = u64_at(take(&mut p, 8)?);
        let slen = u32_at(take(&mut p, 4)?) as usize;
        let source = String::from_utf8_lossy(take(&mut p, slen)?).into_owned();
        let nb = u32_at(take(&mut p, 4)?) as usize;
        let mut blocks = Vec::with_capacity(nb);
        for _ in 0..nb {
            blocks.push(u64_at(take(&mut p, 8)?));
        }
        let nt = u32_at(take(&mut p, 4)?) as usize;
        let mut postings = HashMap::with_capacity(nt);
        for _ in 0..nt {
            let k = u32_at(take(&mut p, 4)?);
            let kind = take(&mut p, 1)?[0];
            if kind == 0 {
                postings.insert(k, Posting::All);
            } else {
                let len = u32_at(take(&mut p, 4)?) as usize;
                let enc = take(&mut p, len)?.to_vec();
                let last = u32_at(take(&mut p, 4)?);
                postings.insert(k, Posting::List { enc, last });
            }
        }
        Ok(TrigramIndex {
            source,
            block_lines,
            last_block_lines,
            indexed_bytes,
            n_lines,
            blocks,
            postings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> String {
        let d =
            std::env::temp_dir().join(format!("geminga-trigram-{}-{}", std::process::id(), name));
        d.to_string_lossy().into_owned()
    }

    fn write_lines(path: &str, lines: &[&str]) {
        let mut f = File::create(path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    fn plain(lines: &[String], pat: &str, ic: bool) -> Vec<u64> {
        lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                if ic {
                    l.to_ascii_lowercase().contains(&pat.to_ascii_lowercase())
                } else {
                    l.contains(pat)
                }
            })
            .map(|(i, _)| i as u64)
            .collect()
    }

    #[test]
    fn finds_lines_and_agrees_with_a_plain_scan() {
        let p = tmp("a.jsonl");
        let mut lines = Vec::new();
        for i in 0..1000 {
            lines.push(format!(
                "{{\"op\":\"load\",\"genes\":[\"G{i}\",\"{}\"],\"window\":[{i},{}]}}",
                if i % 97 == 0 { "GFAP" } else { "SOX2" },
                i + 10
            ));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        write_lines(&p, &refs);
        let idx = TrigramIndex::build(&p, 16).unwrap();
        assert_eq!(idx.n_lines, 1000);
        assert_eq!(idx.n_blocks(), 63);
        assert!(
            idx.n_everywhere() > 10,
            "the JSON keys occur in every block"
        );
        for (pat, ic) in [
            ("gfap", true),
            ("GFAP", false),
            ("gfap", false),
            ("G9", false),
            ("\"op\":\"load\"", true),
            ("G99\"", true),
            ("zzz", true),
            ("", true),
        ] {
            let hits = idx.search(pat, ic).unwrap();
            assert_eq!(
                hits.iter().map(|h| h.0).collect::<Vec<_>>(),
                plain(&lines, pat, ic),
                "{pat} {ic}"
            );
            for (ln, text) in &hits {
                assert_eq!(text, &lines[*ln as usize]);
            }
        }
        // the key trigrams narrow nothing; a gene narrows to its blocks; an absent one to none
        assert!(idx.candidates(b"\"op\":").is_none());
        assert_eq!(idx.candidates(b"gfap").unwrap().len(), 11);
        assert_eq!(idx.candidates(b"zzz").unwrap().len(), 0);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn update_extends_demotes_and_search_covers_the_unindexed_tail() {
        let p = tmp("b.jsonl");
        write_lines(&p, &["alpha one", "beta two alpha", "gamma three"]);
        let mut idx = TrigramIndex::build(&p, 2).unwrap();
        assert_eq!((idx.n_lines, idx.n_blocks()), (3, 2));
        // "alp" was in block 0 only -> block 1 lacks it -> a list, not "everywhere"
        assert_eq!(idx.candidates(b"alpha").unwrap(), vec![0]);
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
            writeln!(f, "delta four alpha").unwrap();
            writeln!(f, "epsilon gamma").unwrap();
            write!(f, "torn gam").unwrap(); // no newline: not a line yet
        }
        // before update: the tail is scanned
        let hits = idx.search("gamma", true).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 4]);
        let added = idx.update().unwrap();
        assert_eq!(added, 2);
        assert_eq!((idx.n_lines, idx.n_blocks()), (5, 3));
        assert_eq!(idx.candidates(b"alpha").unwrap(), vec![0, 1]);
        let hits = idx.search("gamma", true).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 4]);
        assert_eq!(idx.search("ALPHA", true).unwrap().len(), 3);
        // save / load round trip
        let s = tmp("b.tgx");
        idx.save(&s).unwrap();
        let back = TrigramIndex::load(&s).unwrap();
        assert_eq!(back.indexed_bytes, idx.indexed_bytes);
        assert_eq!(back.n_everywhere(), idx.n_everywhere());
        assert_eq!(back.search("delta", true).unwrap().len(), 1);
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&s);
    }

    #[test]
    fn everywhere_marks_survive_partial_block_extension() {
        // block 0 gets "common" in both lines; the extension of the partial block 1 adds a line
        // without "common": block 1 already had it, so it must stay "everywhere"
        let p = tmp("c.jsonl");
        write_lines(&p, &["common a", "common b", "common c"]);
        let mut idx = TrigramIndex::build(&p, 2).unwrap();
        assert!(idx.candidates(b"common").is_none());
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
            writeln!(f, "other d").unwrap(); // completes block 1
            writeln!(f, "other e").unwrap(); // starts block 2, lacking "common"
        }
        idx.update().unwrap();
        assert_eq!(idx.candidates(b"common").unwrap(), vec![0, 1]);
        assert_eq!(idx.search("common", true).unwrap().len(), 3);
        let _ = std::fs::remove_file(&p);
    }
}
