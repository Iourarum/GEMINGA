"""The orchestrator on a real project workbook: InoKAN's cell-culture & organoid dataset table.

    python examples/orchestrate_inokan.py InoKAN_CD_nodes_3_culture_organoids.xlsx [out_dir] [--online]

The workbook has a `Datasets` tab (56 rows of free-text descriptions), a `Nodes` tab (420 CD
designations with HGNC/Ensembl/Entrez/UniProt ids and mouse orthologs), an `ADT_panels` tab
(antibody panels -> genes) and a hand-built `Node_x_Dataset` evidence matrix. The orchestrator

  1. reads the descriptions into pieces and pulls every accession out of the prose,
  2. learns gene identity from `Nodes` (complexes such as CD3 expand to CD3D/E/G) and panels
     from `ADT_panels`,
  3. resolves every piece without transferring a byte — here / one fetch away / behind a DAC /
     paper only — and prices the free routes when online,
  4. recomputes the P/X/F evidence matrix from the descriptions and checks it against the
     hand-built sheet,
  5. writes a blueprint for the InoKAN core panel keyed by Ensembl (what Geneformer wants) and
     builds it — every piece is a hole until files arrive, and the build says so.
"""
import json
import os
import sys

import geminga as g

if len(sys.argv) < 2:
    sys.exit(__doc__)
XLSX = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "/tmp/gem/inokan"
ONLINE = "--online" in sys.argv
os.makedirs(OUT, exist_ok=True)

cat = g.catalog(os.path.join(OUT, "catalog.jsonl"))
lib = g.library(os.path.join(OUT, "library.jsonl"))
orc = g.orchestrator(catalog=cat, library=lib, budget=g.Budget(ram="4GB"))

# 1. describe
pieces = orc.describe(XLSX, sheet="Datasets")
print(f"{len(pieces)} pieces from the Datasets tab; "
      f"{sum(len(p.accessions) for p in pieces)} accessions found in their text\n")

# 2. learn identity and panels
orc.learn_genes(XLSX, sheet="Nodes", symbol="primary_gene", members="human_genes",
                ensembl="ensembl_human (Geneformer key)", entrez="entrez", uniprot="uniprot",
                hgnc="hgnc_id", aliases=("node_id",), group="node_id",
                ortholog="mouse_primary_gene", ortholog_status="mouse_ortholog_status")
orc.learn_panel(XLSX, sheet="ADT_panels", panel="panel", gene="gene_symbol")
gm = orc.harmonizer.genes
print(f"gene map: {len(gm)} records, {len(orc.panels)} antibody panels "
      f"({', '.join(f'{k}: {len(v)}' for k, v in orc.panels.items())})")
print("  CD3 ->", [r.symbol for r in gm.members("CD3")], "| PD-L1 alias known:", "PD-L1" in gm,
      "| CD274 ->", gm.to(["CD274"], key="ensembl")[0], "| mouse:", gm.to(["CD274"], key="ortholog")[0], "\n")

# 3. resolve
orc.resolve(offline=not ONLINE, search_dirs=[OUT])
orc.print_report(limit=100)
print()

# 4. coverage of the core panel, checked against the workbook's own matrix
import openpyxl
wb = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)
node_rows = list(wb["Nodes"].iter_rows(values_only=True))
hdr = node_rows[0]
core = [r[0] for r in node_rows[1:] if r[hdr.index("InoKAN_core")] == "yes"]
cov = orc.coverage(genes=core, key="group")
cov.print(limit=25)
cov.to_csv(os.path.join(OUT, "coverage_core.csv"))

sheet = {}
mx = list(wb["Node_x_Dataset"].iter_rows(values_only=True))
mh = mx[0]
for r in mx[3:]:
    for j, v in enumerate(r[4:], 4):
        if v:
            sheet[(r[0], mh[j])] = "".join(sorted(str(v)))
agree = disagree = 0
diffs = []
for (node, ds), code in sheet.items():
    if node not in core or ds not in orc.pieces:
        continue
    mine = "".join(sorted(cov.code(node, ds).replace("x", "")))
    if mine == code:
        agree += 1
    else:
        disagree += 1
        diffs.append((node, ds, code, mine or "-"))
print(f"\nhand-built Node_x_Dataset vs recomputed, core nodes: {agree} agree, {disagree} differ")
for d in diffs[:12]:
    print(f"  {d[0]:<8} {d[1]:<44} sheet={d[2]:<4} orchestrator={d[3]}")
implied = sum(1 for gname in core for pid in orc.pieces if "x" in cov.code(gname, pid))
print(f"  plus {implied} node×piece cells implied by genome-wide screens (lower-case x), "
      f"which the sheet leaves blank on purpose\n")

# 5. annotate, blueprint, build
ann = orc.annotate()
print(f"catalog: {len(cat.entries())} entries; library: {len(lib.query(intent='need-data'))} need-data flags")
panel = gm.expand(core, key="ensembl")
bp = orc.blueprint("inokan-core-culture", key="ensembl", panel=panel,
                   obs=["sample", "donor", "condition", "perturbation", "guide", "cell_type"],
                   note="InoKAN core CD panel over the culture/organoid datasets, Ensembl-keyed for Geneformer")
bp.save(os.path.join(OUT, "blueprint.json"))
pz = orc.build(bp, os.path.join(OUT, "puzzle"))
pz.print_table()
print(f"\n{pz}: var axis of {pz.n_vars} Ensembl ids from {len(core)} core nodes; "
      f"{len(pz.holes)} holes recorded with their routes in {pz.path}/blueprint.json")
orc.save(os.path.join(OUT, "pieces.json"))
orc.to_csv(os.path.join(OUT, "pieces.csv"))
with open(os.path.join(OUT, "report.json"), "w") as fh:
    json.dump(orc.report(), fh, indent=2)
print(f"\nwrote {OUT}/: pieces.json, pieces.csv, coverage_core.csv, blueprint.json, report.json, puzzle/")
