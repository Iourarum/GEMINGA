# In a wet-lab glioma study — a brainstorm

This is deliberately a brainstorm, not a plan. The honest framing first: GEMINGA is
infrastructure. It will not be the finding in a glioma paper, and a tool-only submission to a
neuro-oncology journal would be misjudged. Where it earns a place is in *how* a study is done
and in the methods section that proves it was done that way.

## Where it fits in a study

**Before sequencing — choosing the comparison atlases.** A glioma project reanalysing public
data faces thirty candidate datasets and a laptop. `acquisition_plan()` prices each route,
`peek()` reads the first reads for kilobytes, `profile()` says what each one probably is and
what would settle it, and the catalog keeps the decisions. The output is a defensible list of
which atlases were integrated and why the rest were not — which reviewers ask about and authors
rarely can document.

**During analysis — the region, not the file.** A tumour-core versus infiltrating-margin
question on a Visium or Xenium section is a window. Two windows, two `Harvest` regions, a
manifest with coordinates, bin level, gene panel and checksums. That manifest *is* the
provenance record for the figure.

**Multi-modal registration.** A snRNA-seq atlas, a Xenium section and a CODEX panel from the
same resection share biology but not coordinates. `StereoSeqDataset`/`VisiumBundle` hold
expression and morphology in one frame; the `Workspace` holds several datasets partitioned and
addressable; `select_frac()` puts them at the same relative position. This is not integration
— it is the disciplined staging that makes integration reproducible.

**Perturbation follow-up.** If the study includes a CRISPRi screen on patient-derived lines,
the guide index (see [the CRISPR page](crispr.md)) picks the cells per target; `pertpy` or
decoupleR does the statistics.

## Concrete glioma questions it could serve

- **Hypoxic niche, three ways.** The inventory has CODEX perinecrotic mapping, MALDI-MSI
  metabolic depletion zones and Raman-aligned Xenium. Windows around the same perinecrotic
  region across modalities, staged once, handed to one analysis. GEMINGA reads two of the three
  today; MSI is the gap.
- **Microtube networks.** Xenium in situ with mIF (Connexin 43). A `MultiChannelTiff` channel
  and a `XeniumBundle` window over the same field, registered by pixel size from
  `experiment.xenium`.
- **Serial biopsy under viral therapy.** Longitudinal, multi-modal. The `Cohort` pattern from
  the MRI work generalises: subject × timepoint × modality, same window across time.
- **Reprogramming and reversion.** The RESYNC and KAIST lines of work want TF-activity
  signatures compared across cell lines and perturbations; the perturbation index is the
  data side of that.

## What a methods section gains

Every figure built through a `Harvest` carries: source file, window, bin or zoom level, gene
selection, row count, SHA-256, and the budget it ran under. "Analysed on a 16 GB workstation
with peak resident memory 4.2 GB" is a sentence a reviewer can check. Resource-aware reanalysis
is itself a legitimate methods contribution — the F1000 / Bioinformatics application-note
format exists for exactly this, and it is the right venue for the *tool*, distinct from the
glioma paper that uses it.

## Publication shapes, honestly

1. **The glioma paper cites GEMINGA in methods** and deposits the manifests as supplementary
   provenance. GEMINGA is not an author claim; reproducibility is.
2. **An application note** — "memory-budgeted access to multi-modal glioma atlases on
   commodity hardware" — with the inventory as the benchmark. Bioinformatics, F1000, or JOSS.
3. **A data-descriptor companion**, if the lab generates new spatial data: the manifest format
   as the release format.

Order 1 first. The tool paper is stronger once a biology paper has used it.

## What has to be true first

- **Real SAW, Space Ranger and Xenium Ranger output has to be tested.** The fixtures are
  synthetic. A version mismatch in an attribute name is the most likely first failure.
- **`fragments.tsv.gz` for scATAC** and **imzML for MSI** are the two readers a glioma
  multi-omics study will miss first.
- **Wet-lab provenance is not GEMINGA's.** Dissociation method, warm-collagenase stress, fresh
  versus frozen — the inventory's own QC checklist — live in the catalog as `annotate()` fields
  the analyst fills in. The tool cannot infer them; it can refuse to let them be forgotten.
