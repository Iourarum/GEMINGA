"""A very small assistant, and a deliberate refusal to bundle one.

GEMINGA does not ship an inference engine. Vendoring one would add gigabytes of model weights,
a compile-time dependency on a CUDA or Metal toolchain, and a maintenance surface larger than
the rest of the library — to a tool whose entire premise is that the user's machine is small.

Instead: a thin adapter over whatever the user already runs. Three backends, tried in order.

``openai``
    Any OpenAI-compatible ``/v1/chat/completions`` endpoint. This is the widest door: it
    covers **mistral.rs** (a pure-Rust, Candle-based engine with an OpenAI-compatible server —
    the natural pairing for a Rust-core project), llama.cpp's ``llama-server``, vLLM, LM Studio
    and hosted APIs, all through one code path.

``ollama``
    ``/api/generate`` on a local Ollama daemon. The path of least resistance for most students,
    and what a laptop with 8 GB can realistically run — a 3B or 4B quantized model answers a
    "what is this protocol" question perfectly well.

``template``
    No model at all. Deterministic, offline, and always available: it formats what the profiler
    already knows and says plainly where a model would have added prose. Used as the fallback
    so that nothing in GEMINGA ever *requires* an LLM to function.

On the names that came up while scoping this: ``mistral.rs`` and ``candle`` are real and
actively developed; ``rustformers/llm`` is archived and should not be built on. ``llmfit``,
``llm-dev`` and ``llm-completion`` did not resolve to maintained projects in that space — if
they are internal or renamed, point this adapter at their endpoint rather than adding a
dependency.

Everything here is optional. Nothing in the core imports it.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["Assistant", "AssistantReply", "assistant", "PROTOCOL_NOTES"]


@dataclass
class AssistantReply:
    text: str
    backend: str
    model: str = ""
    sources: list = field(default_factory=list)
    grounded: bool = True     # False when the model spoke beyond what we gave it

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (f"AssistantReply({self.backend}/{self.model or '-'}, "
                f"{len(self.text)} chars, {len(self.sources)} sources)")


#: Offline fallback prose. Short, checkable, and honest about what it leaves out.
PROTOCOL_NOTES: dict[str, dict] = {
    "scRNA-seq": dict(
        text=("Single-cell RNA sequencing measures transcript abundance one cell at a time. "
              "Tissue is dissociated into a suspension, individual cells are captured — most "
              "often in droplets alongside a barcoded bead — and every transcript from a cell "
              "inherits that cell's barcode plus a unique molecular identifier that lets PCR "
              "duplicates be collapsed. The result is a sparse cells-by-genes count matrix. "
              "Because the cell must survive dissociation, scRNA-seq is biased against fragile "
              "and tightly-embedded cell types, and dissociation itself induces stress-response "
              "genes. Reads are exon-dominated: the captured RNA is mature, spliced cytoplasmic "
              "message, so a typical library shows well under 30% intronic signal. That last "
              "number is the practical way to tell a scRNA-seq library from a single-nucleus "
              "one when the metadata does not say."),
        links=["https://www.ncbi.nlm.nih.gov/pmc/?term=single-cell+RNA-seq+review",
               "https://www.10xgenomics.com/support"]),
    "snRNA-seq": dict(
        text=("Single-nucleus RNA sequencing profiles nuclei rather than whole cells. Nuclei "
              "are released from tissue that may be frozen or archived, which is the main "
              "reason to choose it: samples that cannot be dissociated alive — frozen "
              "biobank tissue, brain, adipose, skeletal muscle — remain usable, and the harsh "
              "dissociation artefacts of scRNA-seq are avoided. The trade is what is being "
              "measured. Nuclear RNA is enriched for nascent, incompletely spliced transcript, "
              "so libraries are intron-rich, often above 50%, and quantification must be run "
              "with intronic reads included or most of the signal is discarded. Cytoplasmic "
              "message is largely absent, so abundance does not match scRNA-seq from the same "
              "tissue and the two should not be merged naively. The file formats are identical "
              "to scRNA-seq, which is exactly why the distinction has to come from metadata or "
              "from the intronic fraction."),
        links=["https://www.ncbi.nlm.nih.gov/pmc/?term=single-nucleus+RNA-seq+comparison"]),
    "Stereo-seq": dict(
        text=("Stereo-seq captures spatially resolved transcriptomes on a chip patterned with "
              "DNA nanoballs at roughly 500 nm pitch, each carrying a coordinate identity "
              "barcode. Tissue is placed on the chip, permeabilised, and released RNA is "
              "captured in place, so every transcript inherits a physical position. Resolution "
              "is therefore sub-cellular in principle; in practice observations are aggregated "
              "into square bins — bin1 up to bin200 — trading resolution for counts per unit. "
              "Chips scale from 1 cm to 13 cm square, which is what makes whole-organ and "
              "whole-embryo sections possible and what makes the data large enough to need "
              "streaming. A paired ssDNA or H&E image registered to the same coordinate frame "
              "supports cell segmentation, making the assay multi-modal by construction."),
        links=["https://en.stomics.tech/", "https://www.biorxiv.org/search/Stereo-seq"]),
    "scATAC-seq": dict(
        text=("Single-cell ATAC sequencing maps open chromatin per cell. A hyperactive Tn5 "
              "transposase inserts sequencing adapters preferentially into accessible DNA, so "
              "read pileups mark regulatory regions rather than transcripts. Features are "
              "genomic intervals — peaks — not gene symbols, which is the quickest way to "
              "recognise the data. Matrices are far sparser and more binary than expression "
              "matrices, and most scRNA-seq analysis defaults are wrong for them."),
        links=["https://www.ncbi.nlm.nih.gov/pmc/?term=scATAC-seq+review"]),
    "bulk RNA-seq": dict(
        text=("Bulk RNA sequencing measures average transcript abundance across all cells in a "
              "sample. It remains the right choice when the question is about condition-level "
              "differences rather than cellular composition: it is cheaper per sample, has far "
              "greater depth per gene, and its statistics are well understood. What it cannot "
              "do is separate a change in expression within a cell type from a change in the "
              "proportion of that cell type, which is the failure mode that pushed the field "
              "toward single-cell and toward deconvolution methods."),
        links=["https://www.ncbi.nlm.nih.gov/pmc/?term=bulk+RNA-seq+differential+expression"]),
}


class Assistant:
    """A small, optional assistant for explaining protocols and summarising sources.

        ai = g.assistant()                       # auto-detects a local backend
        print(ai.explain_protocol("snRNA-seq"))
        print(ai.describe(profile))
    """

    def __init__(self, backend: str = "auto", model: str = "",
                 base_url: str = "", api_key: str = "", timeout: float = 60.0,
                 max_words: int = 300):
        self.timeout = timeout
        self.max_words = max_words
        self.api_key = api_key or os.environ.get("GEMINGA_LLM_KEY", "")
        self.backend, self.base_url, self.model = self._resolve(backend, base_url, model)

    # ---- backend discovery ----
    def _resolve(self, backend: str, base_url: str, model: str):
        env_url = os.environ.get("GEMINGA_LLM_URL", "")
        env_model = os.environ.get("GEMINGA_LLM_MODEL", "")
        if backend == "template":
            return "template", "", ""
        if backend in ("openai", "ollama"):
            url = base_url or env_url or (
                "http://localhost:11434" if backend == "ollama" else "http://localhost:8080")
            return backend, url.rstrip("/"), model or env_model or "local"
        # auto: try the endpoints a student is likely to already be running
        for kind, url, probe in (
            ("openai", base_url or env_url or "http://localhost:8080", "/v1/models"),
            ("openai", "http://localhost:1234", "/v1/models"),      # LM Studio
            ("ollama", "http://localhost:11434", "/api/tags"),
        ):
            if not url:
                continue
            got = self._probe(url.rstrip("/") + probe)
            if got is not None:
                return kind, url.rstrip("/"), model or env_model or self._pick(got) or "local"
        return "template", "", ""

    def _probe(self, url: str):
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=2.5) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            return None

    @staticmethod
    def _pick(payload) -> str:
        for key in ("data", "models"):
            items = payload.get(key) if isinstance(payload, dict) else None
            if items:
                first = items[0]
                return first.get("id") or first.get("name") or ""
        return ""

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": "geminga/0.4"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @property
    def available(self) -> bool:
        return self.backend != "template"

    # ---- generation ----
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, headers=self._headers(),
                                     data=json.dumps(payload).encode())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def ask(self, prompt: str, system: str = "", max_words: Optional[int] = None
            ) -> AssistantReply:
        """One turn. Falls back to `template` prose when no backend is reachable."""
        n = max_words or self.max_words
        system = system or (
            "You are a concise bioinformatics reference. Answer in plain prose, no lists. "
            f"Stay under {n} words. If you are unsure, say so rather than guessing. "
            "Never invent citations, accession numbers or URLs.")
        try:
            if self.backend == "openai":
                d = self._post("/v1/chat/completions", {
                    "model": self.model, "temperature": 0.2, "max_tokens": int(n * 2),
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}]})
                return AssistantReply(d["choices"][0]["message"]["content"].strip(),
                                      self.backend, self.model, grounded=False)
            if self.backend == "ollama":
                d = self._post("/api/generate", {
                    "model": self.model, "prompt": f"{system}\n\n{prompt}", "stream": False,
                    "options": {"temperature": 0.2, "num_predict": int(n * 2)}})
                return AssistantReply(d.get("response", "").strip(), self.backend,
                                      self.model, grounded=False)
        except Exception as e:
            return AssistantReply(
                f"(no model reachable: {type(e).__name__}) " + self._template(prompt),
                "template", grounded=True)
        return AssistantReply(self._template(prompt), "template", grounded=True)

    def _template(self, prompt: str) -> str:
        for label, spec in PROTOCOL_NOTES.items():
            if label.lower() in prompt.lower():
                return spec["text"]
        return ("No local model is reachable and this protocol has no offline note. Start an "
                "OpenAI-compatible server (mistral.rs, llama.cpp, LM Studio) or Ollama, or set "
                "GEMINGA_LLM_URL, and ask again.")

    # ---- the two things GEMINGA actually wants from it ----
    def explain_protocol(self, assay: str, context: str = "",
                         max_words: int = 300) -> AssistantReply:
        """Roughly 300 words on a sequencing protocol, plus places to read further."""
        note = PROTOCOL_NOTES.get(assay)
        links = list(note["links"]) if note else []
        if not self.available:
            r = AssistantReply(note["text"] if note else self._template(assay),
                               "template", grounded=True)
            r.sources = links
            return r
        prompt = (f"Explain the {assay} protocol for a bioinformatics graduate student in about "
                  f"{max_words} words: what is physically measured, what the main biases and "
                  f"failure modes are, and how it differs from the assay it is most often "
                  f"confused with. Plain prose."
                  + (f"\n\nContext from the dataset at hand: {context}" if context else ""))
        r = self.ask(prompt, max_words=max_words)
        r.sources = links
        return r

    def describe(self, prof, max_words: int = 200) -> AssistantReply:
        """Summarise a Profile, and say what remains unresolved."""
        facts = json.dumps(prof.to_dict(), default=str)[:3000]
        if not self.available:
            lines = [prof.explain()]
            if prof.best and not prof.confident:
                lines.append("\nNo model reachable, so this is the profiler's own reading.")
            return AssistantReply("\n".join(lines), "template", grounded=True)
        prompt = (f"Here is an automated profile of a dataset:\n{facts}\n\n"
                  f"In under {max_words} words, say what this dataset most likely is, what the "
                  f"evidence rests on, and — most importantly — what remains genuinely "
                  f"uncertain and what observation would resolve it. Do not overstate "
                  f"confidence beyond the probabilities given.")
        return self.ask(prompt, max_words=max_words)

    def search_notes(self, query: str, max_words: int = 250) -> AssistantReply:
        """Ask for orientation plus places to look. Links are not fetched or verified here —
        treat them as leads, not citations."""
        r = self.ask(
            f"{query}\n\nGive a short orientation, then name the kinds of source worth "
            f"consulting (specific journals, consortium portals, tool documentation). "
            f"Do not fabricate URLs or accession numbers.", max_words=max_words)
        r.sources = [u for u in re.findall(r"https?://[^\s)\]]+", r.text)]
        return r

    def __repr__(self) -> str:
        return (f"Assistant({self.backend}"
                f"{'/' + self.model if self.model else ''}"
                f"{', offline fallback' if not self.available else ''})")


def assistant(backend: str = "auto", **kw) -> Assistant:
    """A small optional assistant. Detects a local OpenAI-compatible or Ollama endpoint;
    falls back to offline template prose so nothing ever hard-depends on a model."""
    return Assistant(backend=backend, **kw)
