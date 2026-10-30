# The orchestrator

*Gh'è minga* — "it is not there" — is what a Milanese astronomer said of the pulsar this
library is named after: for years there was nothing at its position in the optical sky. It is
also the state of most datasets a project works with. They exist as a row in a spreadsheet — a
first author, an accession in the notes, a modality column, "raw reads under dbGaP" — and the
project is a puzzle assembled from such rows, some pieces on disk, some one download away, some
behind a Data Access Committee, some only a paper.

The orchestrator treats a description as a **piece** and a project as a **puzzle**, and gives
four verbs that never move a byte they were not asked to:

| verb | what it does | what it writes down |
|---|---|---|
| **describe** | rows of any inventory → `Piece` objects with accessions pulled from the prose, facets harmonised, genes by role | the verbatim source row, file and line |
| **resolve** | here / one fetch away (streamable?) / behind a DAC / paper only / unresolved, with the free route and its price | the route that earned the status, every controlled component |
| **harmonise + annotate** | facets into the catalog vocabulary, genes into one id space, obs columns onto canonical names; catalog records for every piece, `need-data` flags for every hole | the mapping applied and what it could not place |
| **build** | a `Blueprint` executed under the budget into a `Harvest`: local pieces aligned on one feature axis, holes recorded, never fatal | `blueprint.json` — provenance per piece, alignment loss, budget peak, the holes and their routes |

```{image} _static/orchestrator.png
:alt: 56 described pieces resolved without transfer; the node × dataset evidence matrix recomputed from the rows; the accessions found in the prose
:class: gem-figure
```

## Describe

```python
orc = g.orchestrator(catalog=g.catalog("~/inokan/catalog.jsonl"),
                     library=g.library("~/inokan/library.jsonl"),
                     budget=g.Budget(ram="8GB"))
pieces = orc.describe("InoKAN_CD_nodes_3_culture_organoids.xlsx", sheet="Datasets")
```

`describe()` reads CSV/TSV, a workbook tab (`sheet=None` picks the tab whose cells hold the
most accessions), Parquet, JSONL, a DataFrame or a list of dicts. Column names are matched
against a synonym table that has seen four inventories with four vocabularies — `dataset_id`,
`Dataset / Cohort`, `Dataset Focus`; `accession`, `Accession(s)`, `Accession / Host`;
`modalities`, `Protocols Used`, `sequences_modalities` — and `columns=` pins anything the
guesser gets wrong. Accessions are pulled from **every** text cell, so a series mentioned only in
the notes still resolves:

```python
g.find_accessions("DepMap 24Q4 (Figshare+ 27993248); dbGaP phs003444.v1.p1 for sequence data")
# [phs003444.v1.p1 (dbgap), 10.6084/m9.figshare.27993248 (figshare_doi)]
```

GEO, SRA/ENA, BioProject, BioSample, EGA, dbGaP, ArrayExpress, GSA, CNGB, STOmics, Single Cell
Portal, DUOS, OpenNeuro, Zenodo (record numbers become DOIs), figshare, Mendeley, bare DOIs and
URLs. A URL that only carries an accession already found — a GEO query page — is dropped.

A `Piece` carries `facets` in the catalog's fixed key set, `modalities` as canonical assay
labels, `perturbation` (type, scope, count), `genes` by role — `perturbed`, `protein`, `focus` —
`expected` counts read out of prose ("58,302 profiled; 19,834 after QC" → 58 302; "~1.7M cells"
→ 1 700 000, and "100 MS patients" is not 100 million), `access`, `formats`, and the row itself.

## Resolve

```python
orc.resolve(search_dirs=["/mnt/nas/inokan"])     # offline by default
orc.print_report()
```

```
gh'è minga — 56 piece(s) described from 1 source(s)
  here                   0   0 B on disk
  one fetch away        41   35 streamable · sizes known for 0 (0 B) · metered ~$0.00
  behind a DAC           0   -   (+4 open pieces with a controlled-access component)
  paper only            15   landing page, DOI or URL — no file to price
  unresolved             0   nothing accession-shaped in the row

  id                                       assay                       cells  status      via
  datlinger2017_cropseq_jurkat             scRNA-seq                   5,900  streamable  GSE92872
  depmap_24q4_crispr_chronos               pooled CRISPR screen            -  streamable  10.6084/m9.figshare.27993248
  li2023_choose_organoid_crispr            scRNA-seq                  49,754  streamable  E-MTAB-13148
  …
```

Local first: the catalog's `local_paths`, a `Vault`'s members, and any `search_dirs` are checked
for the piece id or an accession in a file name; `orc.attach(piece, path)` is how the analyst says
"this file is that piece" when names do not match, and it records the fact in the catalog. Then
every accession goes through `geminga.acquire.plan()`. A piece gets the *best* status any of its
accessions earns, and `Resolution.controlled` keeps the ones that would need approval — "processed
matrices on GEO, raw reads under EGA" is one piece with both. Offline, SRA-family accessions are
reported *fetchable, size unknown* because only ENA's file report can say more; `resolve(offline=False)`
asks ENA, GEO, Zenodo and figshare for sizes and Range support.

A row that names a host but no record — "on figshare" — is *unresolved* with a note saying which
host to search, rather than silently the same as a row that says nothing.

## Harmonise

Three small tools, one wrapper, all in `geminga.harmonize`:

**`Vocabulary`** — controlled facet values with extensible synonyms. `assays("scRNA-seq; CROP-seq;
10x 3' (CRISPRi)")` → `["scRNA-seq", "CROP-seq"]`; `primary_assay()` returns the measurement, not
the perturbation wrapper (CROP-seq *is* scRNA-seq with guides); `perturbation()` → `CRISPRko` /
`CRISPRi` / `CRISPRa` / `base editing` / `ORF overexpression` / `chemical`; `organism("human")` →
`Homo sapiens`; `access("open matrices, raw under dbGaP")` → `controlled`. Short synonyms must
be whole words, so `wes` does not fire inside "western". `vocab.add("assay", "TAP-seq", [...])`
extends any table.

**`GeneMap`** — one gene, many names. Built from any table with some of symbol, Ensembl, Entrez,
UniProt, aliases, orthologs — an HGNC dump (`GeneMap.from_hgnc`), a 10x `features.tsv`
(`from_features`), or a project's node sheet (`from_table` with `members=` for complexes: CD3
expands to CD3D, CD3E, CD3G, and `group=` for the designation that names them). Then:

```python
gm.to(["CD274", "PD-L1", "ENSG00000120217.12", "29126"], key="ensembl")   # one id space
gm.expand(["CD3"], key="ensembl")                                          # complexes → members
gm.canonical("CD3E", "group")                                              # 'CD3'
aln = gm.align({"a": var_a, "b": var_b}, key="ensembl", how="intersection", panel=cd_genes)
aln.explain()          # per piece: in / mapped / kept / unmapped, with examples
```

An alignment is the shared feature axis of several matrices plus, per piece, where each shared
feature sits in the source and how many features were lost getting there. `panel=` makes the axis
a gene list in the list's order — what a node-centred project wants — and `how="union"` zero-fills
instead of intersecting. A name the map cannot place is reported, never guessed: the InoKAN
`Nodes` sheet knows `CD274` but not `PD-1`, and the coverage report says so.

**`ColumnMap`** — `patient_id` is a donor, `sgRNA` is a guide, `leiden` is a cluster,
`orig.ident` is a sample. `guess(columns)` returns the mapping, `fixed=` pins what you know.

## Annotate and cover

```python
orc.annotate()            # catalog: facets, accessions, status per piece; library: a need-data flag per hole
cov = orc.coverage(genes=core_nodes, key="group")
cov.print()
```

Coverage is a gene × piece matrix of evidence codes — **P** protein measured (from the listed
targets or an antibody panel learned with `learn_panel()`), **X** perturbed, **x** implied by a
genome-wide screen, **F** named as a focus, **R** present in a local matrix after harmonisation
(`measured=puzzle.measured()`). On the InoKAN workbook the recomputed matrix agrees with the
hand-built `Node_x_Dataset` sheet in 159 of 164 coded cells; the five differences are two names
the sheet's author resolved by hand (`CD8` → CD8a, `PD-1` → CD279), which the report lists as
unplaceable, and two cells where the workbook chose not to mark genome-wide perturbations that
the rows nonetheless list.

Every hole becomes a `need-data` flag in the library carrying the piece's genes, accessions,
status and route, so `Library.leads()` and `related()` treat "the dataset we do not have yet" like
any other thing that needs more data. Running `annotate()` again updates those flags rather than
duplicating them, and a piece that has arrived since (`attach()`, or a file that appeared in a
`search_dirs` directory) has its flag resolved.

## Build

```python
bp = orc.blueprint("inokan-core", key="ensembl", panel=gm.expand(core, key="ensembl"),
                   obs=["sample", "donor", "condition", "perturbation", "guide", "cell_type"])
puzzle = orc.build(bp, "~/handoff/inokan-core")
puzzle.print_table()
```

```
puzzle inokan-core at /home/s/handoff/inokan-core
  3 piece(s) built, 2 hole(s); 500 obs x 71 vars keyed by ensembl (panel of 71)
  + A_jurkat        300 obs   4/8 vars kept  h5ad     7.5 KB peak
  + B_k562           80 obs   4/5 vars kept  10x-h5   1.9 KB peak
  + C_thp1          120 obs   4/6 vars kept  mtx      4.2 KB peak
  - D_missing       behind a DAC  via ega
  - E_paper         paper only   via doi
```

A build opens each local piece with the reader its container needs — `.h5ad` (CSR row blocks),
Cell Ranger `.h5` (one feature type, `RNA` by default), an mtx triple (COO stream, restricted to
the panel on the way through) — aligns the feature axis, streams the cells under the budget, and
writes a `Harvest`: `X_<piece>.parquet` as COO (`cell`, `gene`, `value`) against a shared
`obs.parquet` (cell id, piece, the canonical columns found, every facet) and `var.parquet` (every
id the map knows for each feature, and an `in_<piece>` column per piece). Pieces that are tables
rather than matrices — a screen's hit list, a gene-effect matrix — are streamed to
`T_<piece>.parquet` with their gene column mapped into the key space (long form) or their gene
columns renamed (wide form, original headers kept in the Parquet metadata). `puzzle.to_anndata()`
assembles the matrices in memory when they fit.

`blueprint.json` beside the manifest is the provenance: the blueprint, the harmoniser's log, the
alignment report, and per piece the source row, accessions, local path and checksum, reader,
chunks, obs-column mapping, features in / mapped / kept with unmapped examples, and peak resident
bytes; per hole the status, route, size and price. A build with 56 holes and no pieces is a
valid, honest artifact — it is the shopping list.

## What it does not do

It does not download (`acquire` builds the command; a download tool moves the bytes; the next
`resolve()` notices). It does not integrate — no batch correction, no embedding — a harmonised,
aligned, annotated matrix with its provenance is where PRISM, HyVESTA or scanpy start. It does not
guess identities: an unplaceable name stays unplaceable and is reported. And it does not read
spatial pieces into a build yet; a Stereo-seq or Visium piece resolves, annotates and counts
toward coverage, and its regions go through `Harvest.add(view)` as before.
