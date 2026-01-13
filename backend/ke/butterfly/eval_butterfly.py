"""
ButterflyKE-style evaluation for Knowledge Editing.

Implements:
1) Next-hop relation extraction (1-hop neighbors around subject/object)
2) Inherent relation induction (inverse / symmetric / transitive) using relation metadata
3) Butterfly Index (BI) exactly as defined: mean over neighbors of (I[orig correct] - I[edited correct])

Requirements:
- A "KnowledgeGraph" backend that can return 1-hop triples and labels for entities/relations.
- An LLM (transformers) to answer QA prompts derived from triples.
- Optional: apply edits via MEMIT or ROME using your project hooks.

All comments/strings are in English by design.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Protocol

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# Text normalization & model answering
# =========================================================

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s\-]")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize(text: str) -> str:
    s = (text or "").strip()
    s = s.split("\n")[0].strip()
    s = _PUNCT.sub("", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def id_to_text(x: str) -> str:
    """Convert KG-style ids like 'E:Quebec_City' to human-readable text."""
    if x is None:
        return ""
    if ":" in x:
        x = x.split(":", 1)[1]
    x = x.replace("_", " ")
    return x.strip()


def safe_entity_label(kg, entity_id: str) -> str:
    """Return a human-readable label for an entity id, falling back to id_to_text."""
    lab = kg.label_entity(entity_id)
    if not lab or lab == entity_id:
        return id_to_text(entity_id)
    # If labels are also KG ids, sanitize again
    if ":" in lab:
        return id_to_text(lab)
    return lab.strip()


def safe_relation_label(kg, rel_id: str) -> str:
    """Return a human-readable label for a relation id, falling back to id_to_text."""
    lab = kg.label_relation(rel_id)
    if not lab or lab == rel_id:
        return id_to_text(rel_id)
    if ":" in lab:
        return id_to_text(lab)
    return lab.strip()

def greedy_answer(
    model,
    tok,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 12,
) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)

    # Best-effort: keep suffix after prompt
    suffix = full[len(prompt):] if full.startswith(prompt) else full
    suffix = suffix.strip().split("\n")[0].strip()
    suffix = re.split(r"[\.!,;:]", suffix)[0]
    return suffix.strip()


# =========================================================
# Core data structures
# =========================================================

@dataclass(frozen=True)
class Triple:
    """
    A knowledge graph triple.
    Use ids for subject/predicate/object; labels can be retrieved from KG.
    """
    s: str
    p: str
    o: str

@dataclass
class QAProbe:
    """
    A QA probe derived from a triple.
    q: question string
    g: gold answer string (ground truth)
    triple: the originating triple
    """
    q: str
    g: str
    triple: Triple

@dataclass
class BIItemResult:
    triple: Triple
    question: str
    gold: str
    pre_answer: str
    post_answer: str
    pre_correct: bool
    post_correct: bool

@dataclass
class ButterflyReport:
    method: str                   # "none" | "memit" | "rome"
    model_name: str
    device: str
    edit_triple: Triple           # (s, p, o_new) as edited target representation
    original_object: str          # o_old id
    n_probes: int
    butterfly_index: float
    pre_accuracy: float
    post_accuracy: float
    details: List[BIItemResult]


# =========================================================
# Knowledge graph interface (pluggable)
# =========================================================

class KnowledgeGraph(Protocol):
    """
    Minimal KG protocol for ButterflyKE measurement.
    """

    def label_entity(self, entity_id: str) -> str:
        ...

    def label_relation(self, rel_id: str) -> str:
        ...

    def one_hop(self, entity_id: str, *, limit: int = 200) -> List[Triple]:
        """
        Return 1-hop triples containing entity_id either as subject or object.
        """
        ...

    def get_object(self, s: str, p: str) -> Optional[str]:
        """
        Return an object id for (s, p, ?) if available, else None.
        Useful to retrieve o_old given (s, p).
        """
        ...


# =========================================================
# Local JSONL KG backend (offline, reproducible)
# =========================================================

class LocalJsonlKG:
    """
    Offline KG backend from a JSONL file with records like:
    {"s":"E:Canada","p":"R:capital","o":"E:Ottawa","s_label":"Canada","p_label":"capital of","o_label":"Ottawa"}

    - Entity and relation ids are arbitrary strings, but must match across triples.
    - Labels are optional per triple; the backend will build maps.

    This is the easiest way to get "true BI" without any online dependency.
    """

    def __init__(self, path_jsonl: str):
        self.path = path_jsonl
        self.triples: List[Triple] = []
        self.ent_label: Dict[str, str] = {}
        self.rel_label: Dict[str, str] = {}
        self.by_entity: Dict[str, List[int]] = {}   # entity_id -> indices in self.triples
        self.by_sp: Dict[Tuple[str, str], str] = {} # (s,p) -> o

        self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                s = rec["s"]
                p = rec["p"]
                o = rec["o"]
                t = Triple(s=s, p=p, o=o)
                idx = len(self.triples)
                self.triples.append(t)

                # Labels (best-effort)
                if "s_label" in rec:
                    self.ent_label.setdefault(s, rec["s_label"])
                if "o_label" in rec:
                    self.ent_label.setdefault(o, rec["o_label"])
                if "p_label" in rec:
                    self.rel_label.setdefault(p, rec["p_label"])

                # Index for 1-hop lookup
                self.by_entity.setdefault(s, []).append(idx)
                self.by_entity.setdefault(o, []).append(idx)

                # Simple SPO lookup (first wins)
                self.by_sp.setdefault((s, p), o)

    def label_entity(self, entity_id: str) -> str:
        return self.ent_label.get(entity_id, entity_id)

    def label_relation(self, rel_id: str) -> str:
        return self.rel_label.get(rel_id, rel_id)

    def one_hop(self, entity_id: str, *, limit: int = 200) -> List[Triple]:
        idxs = self.by_entity.get(entity_id, [])
        out = [self.triples[i] for i in idxs[:limit]]
        return out

    def get_object(self, s: str, p: str) -> Optional[str]:
        return self.by_sp.get((s, p))


# =========================================================
# Relation metadata for induction (inverse/symmetric/transitive)
# =========================================================

@dataclass
class RelationMeta:
    """
    relation_id: the predicate id in KG
    inverse_of: predicate id that is inverse (optional)
    symmetric: whether (a R b) implies (b R a)
    transitive: whether (a R b) and (b R c) implies (a R c)
    """
    relation_id: str
    inverse_of: Optional[str] = None
    symmetric: bool = False
    transitive: bool = False


class RelationInducer:
    """
    Implements inherent relation induction.
    Given:
    - next-hop triples
    - relation metadata
    Generates additional induced triples.
    """

    def __init__(self, rel_meta: Dict[str, RelationMeta]):
        self.rel_meta = rel_meta

    def induce_inverse_and_symmetric(self, triples: Iterable[Triple]) -> List[Triple]:
        induced: List[Triple] = []
        seen: Set[Triple] = set()

        for t in triples:
            meta = self.rel_meta.get(t.p)
            if not meta:
                continue

            # Inverse: (s,p,o) -> (o, inv(p), s)
            if meta.inverse_of:
                it = Triple(s=t.o, p=meta.inverse_of, o=t.s)
                if it not in seen:
                    induced.append(it)
                    seen.add(it)

            # Symmetric: (s,p,o) -> (o,p,s)
            if meta.symmetric:
                it = Triple(s=t.o, p=t.p, o=t.s)
                if it not in seen:
                    induced.append(it)
                    seen.add(it)

        return induced

    def induce_transitive(self, kg: KnowledgeGraph, triples: Iterable[Triple], *, limit_per_node: int = 200) -> List[Triple]:
        """
        Transitive closure of length 2 only (as a practical induction):
        If (a R b) and (b R c) then (a R c) for transitive R.

        This matches typical "inherent relation induction" behavior in practice:
        a single extra hop to create implied facts.
        """
        induced: List[Triple] = []
        seen: Set[Triple] = set()

        # Group by predicate to avoid unnecessary work
        by_p: Dict[str, List[Triple]] = {}
        for t in triples:
            by_p.setdefault(t.p, []).append(t)

        for p, tlist in by_p.items():
            meta = self.rel_meta.get(p)
            if not meta or not meta.transitive:
                continue

            # For each (a R b), find (b R c) in 1-hop around b, then add (a R c)
            for t in tlist:
                b = t.o
                neighbors = kg.one_hop(b, limit=limit_per_node)
                for nb in neighbors:
                    if nb.p != p:
                        continue
                    # nb: (b R c) or (c R b) depending on direction; keep the forward case only
                    if nb.s != b:
                        continue
                    c = nb.o
                    it = Triple(s=t.s, p=p, o=c)
                    if it not in seen and it != t:
                        induced.append(it)
                        seen.add(it)

        return induced


# =========================================================
# Triple -> QA conversion (KG-based analysis)
# =========================================================


class QAGenerator:
    """
    Converts a triple into a QA probe using relation-aware question templates.

    This improves pre-edit neighborhood accuracy for weaker LMs (e.g., GPT-2),
    because generic patterns like "What is the located in of Toronto?" are unnatural.
    """

    # Predicate ids must match your mini-KG
    TEMPLATE_BY_PREDICATE = {
        "R:capital": ("What is the capital of {S}?", "O"),
        "R:currency": ("What is the currency of {S}?", "O"),
        "R:continent": ("On which continent is {S} located?", "O"),
        "R:official_language": ("What is an official language of {S}?", "O"),
        "R:government_type": ("What type of government does {S} have?", "O"),
        "R:head_of_state": ("Who is the head of state of {S}?", "O"),
        "R:country_code": ("What is the country code of {S}?", "O"),
        "R:largest_city": ("What is the largest city of {S}?", "O"),
        "R:located_in": ("Where is {S} located?", "O"),
        "R:time_zone": ("What time zone is {S} in?", "O"),
        "R:iso_code": ("What is the ISO code of {S}?", "O"),
        "R:symbol": ("What is the symbol of {S}?", "O"),
        # Optional / synthetic; keep only if it helps
        "R:province_or_state": ("What province/state is {S} in?", "O"),
    }

    # If a predicate is not in this whitelist, we skip it to avoid noisy probes.
    PREDICATE_WHITELIST = set(TEMPLATE_BY_PREDICATE.keys())

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def triple_to_probe(self, t: Triple) -> QAProbe:
        # Skip unrecognized predicates to keep probes high-quality
        if t.p not in self.PREDICATE_WHITELIST:
            # Raise to allow caller to skip cleanly
            raise ValueError(f"Unsupported predicate for QA probe: {t.p}")

        s_lab = safe_entity_label(self.kg, t.s)
        o_lab = safe_entity_label(self.kg, t.o)

        template, gold_side = self.TEMPLATE_BY_PREDICATE[t.p]
        q = template.format(S=s_lab)

        # Gold is always object label in our setting
        g = o_lab
        return QAProbe(q=q, g=g, triple=t)

    def probe_prompt(self, probe: QAProbe) -> str:
        return f"Q: {probe.q}\nA:"


class NextHopExtractor:
    """
    Extracts next-hop relations around the edit triple.
    Typically includes:
    - 1-hop neighbors of subject s
    - 1-hop neighbors of object o_old (or o_new depending on design)
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def extract(self, s: str, o: str, *, limit: int = 200) -> List[Triple]:
        ns = self.kg.one_hop(s, limit=limit)
        no = self.kg.one_hop(o, limit=limit)

        # Deduplicate
        seen: Set[Triple] = set()
        out: List[Triple] = []
        for t in ns + no:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out


# =========================================================
# Butterfly measurement (ButterflyKE step 3)
# =========================================================

class ButterflyMeasurer:
    """
    Computes BI and accuracy pre/post using QA probes derived from triples.
    """

    def __init__(
        self,
        *,
        kg: KnowledgeGraph,
        qa_generator: QAGenerator,
        device: torch.device,
        max_new_tokens: int = 12,
    ):
        self.kg = kg
        self.qa = qa_generator
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _is_correct(self, pred: str, gold: str) -> bool:
        return normalize(pred) == normalize(gold) or normalize(gold) in normalize(pred)

    def measure(
        self,
        model_orig,
        model_edit,
        tok,
        probes: List[QAProbe],
    ) -> Tuple[float, float, float, List[BIItemResult]]:
        """
        Returns:
        - BI
        - pre accuracy
        - post accuracy
        - per-item details
        """
        details: List[BIItemResult] = []
        pre_hits = 0
        post_hits = 0

        for pr in probes:
            prompt = self.qa.probe_prompt(pr)

            a_pre = greedy_answer(model_orig, tok, prompt, device=self.device, max_new_tokens=self.max_new_tokens)
            a_post = greedy_answer(model_edit, tok, prompt, device=self.device, max_new_tokens=self.max_new_tokens)

            pre_ok = self._is_correct(a_pre, pr.g)
            post_ok = self._is_correct(a_post, pr.g)

            pre_hits += int(pre_ok)
            post_hits += int(post_ok)

            details.append(BIItemResult(
                triple=pr.triple,
                question=pr.q,
                gold=pr.g,
                pre_answer=a_pre,
                post_answer=a_post,
                pre_correct=pre_ok,
                post_correct=post_ok,
            ))

        n = max(1, len(probes))
        pre_acc = pre_hits / n
        post_acc = post_hits / n

        # BI = mean( I[orig correct] - I[edited correct] )
        bi = sum((1 if d.pre_correct else 0) - (1 if d.post_correct else 0) for d in details) / n
        return bi, pre_acc, post_acc, details


# =========================================================
# Editing hooks (MEMIT/ROME) - integrate with your project
# =========================================================

def apply_edit_none(model):
    return model

def apply_edit_memit(model, tok, *, requests: List[Dict[str, Any]]):
    """
    Project hook: replace import paths if needed.

    Expected request format (typical MEMIT):
    {
      "prompt": "The capital of {} is",
      "subject": "Canada",
      "target_new": {"str": " Toronto"},
      ...
    }
    """
    from backend.ke.memit.memit_main import apply_memit_to_model
    from backend.ke.memit_config import MemitConfig
    from backend.ke.memit.memit_hparams import MEMITHyperParams

    memit_cfg = MemitConfig.for_gpt2_xl()
    hparams = MEMITHyperParams.from_memit_config(memit_cfg)

    model_edit, _ = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=requests,
        hparams=hparams,
        copy_model=True,
        return_orig_weights=False,
    )
    return model_edit

def apply_edit_rome(model, tok, *, subject: str, prompt: str, new_object: str):
    """
    Project hook: replace import paths if needed.
    """
    from ke.rome.engine import edit_fact
    from ke.rome.config import RomeConfig

    config = RomeConfig.for_gpt2_xl()
    model_edit, _ = edit_fact(
        model=model,
        tok=tok,
        subject=subject,
        prompt=prompt,
        new_object=new_object,
        config=config,
        copy_model=True,
        return_orig_weights=False,
    )
    return model_edit


# =========================================================
# End-to-end runner
# =========================================================

class ButterflyKERunner:
    """
    End-to-end pipeline consistent with ButterflyKE:
    - identify original triple (s,p,o_old)
    - next-hop extraction around (s,o_old)
    - induction (inverse/symmetric/transitive)
    - QA probes -> BI computation
    - apply edit with MEMIT or ROME
    """

    def __init__(
        self,
        *,
        kg: KnowledgeGraph,
        rel_meta: Dict[str, RelationMeta],
        model_name: str,
        device_str: Optional[str] = None,
        max_new_tokens: int = 12,
    ):
        self.kg = kg
        self.rel_inducer = RelationInducer(rel_meta)
        self.model_name = model_name
        self.device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model_orig = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model_orig.eval()

        self.qa_gen = QAGenerator(kg)
        self.extractor = NextHopExtractor(kg)
        self.measurer = ButterflyMeasurer(
            kg=kg,
            qa_generator=self.qa_gen,
            device=self.device,
            max_new_tokens=max_new_tokens,
        )

    def build_probes(
        self,
        *,
        s: str,
        o_old: str,
        neighbor_limit: int = 200,
        max_probes: int = 400,
        include_induction: bool = True,
    ) -> List[QAProbe]:
        # Step 1: next-hop extraction
        neighbors = self.extractor.extract(s, o_old, limit=neighbor_limit)

        # Step 2: inherent relation induction
        induced: List[Triple] = []
        if include_induction:
            induced += self.rel_inducer.induce_inverse_and_symmetric(neighbors)
            induced += self.rel_inducer.induce_transitive(self.kg, neighbors, limit_per_node=neighbor_limit)

        all_triples: List[Triple] = []
        seen: Set[Triple] = set()
        for t in neighbors + induced:
            if t not in seen:
                all_triples.append(t)
                seen.add(t)

        # Convert to probes
        probes: List[QAProbe] = []
        for t in all_triples[:max_probes]:
            try:
                pr = self.qa_gen.triple_to_probe(t)
            except ValueError:
                # Skip predicates that do not have a reliable QA template
                continue
            # Skip empty golds
            if normalize(pr.g) == "":
                continue
            probes.append(pr)

        return probes

    def run(
        self,
        *,
        s: str,
        p: str,
        o_new: str,
        method: str = "memit",           # "none" | "memit" | "rome"
        neighbor_limit: int = 200,
        max_probes: int = 400,
        include_induction: bool = True,
        out_path: Optional[str] = None,
    ) -> ButterflyReport:
        # Retrieve original object from KG
        o_old = self.kg.get_object(s, p)
        if o_old is None:
            raise ValueError("Original object (o_old) not found in KG for (s,p). Provide a KG triple for it first.")

        probes = self.build_probes(
            s=s,
            o_old=o_old,
            neighbor_limit=neighbor_limit,
            max_probes=max_probes,
            include_induction=include_induction,
        )

        # Apply edit
        if method == "none":
            model_edit = apply_edit_none(self.model_orig)
        elif method == "memit":
            # Build a MEMIT-like request; you can extend it with case_id/title/etc.
            s_lab = safe_entity_label(self.kg, s)
            o_lab_new = safe_entity_label(self.kg, o_new)
            print(f"[Edit] subject='{s_lab}' new_object='{o_lab_new}'")
            request = {
                "prompt": "The capital of {} is",
                "subject": s_lab,
                "target_new": {"str": f" {o_lab_new}"},
            }
            model_edit = apply_edit_memit(self.model_orig, self.tok, requests=[request])
        elif method == "rome":
            s_lab = safe_entity_label(self.kg, s)
            o_lab_new = safe_entity_label(self.kg, o_new)
            model_edit = apply_edit_rome(
                self.model_orig,
                self.tok,
                subject=s_lab,
                prompt="The capital of {} is",
                new_object=o_lab_new,
            )
        else:
            raise ValueError("method must be one of: 'none', 'memit', 'rome'")

        model_edit.eval()

        # Step 3: BI measurement
        bi, pre_acc, post_acc, details = self.measurer.measure(
            self.model_orig, model_edit, self.tok, probes
        )

        report = ButterflyReport(
            method=method,
            model_name=self.model_name,
            device=str(self.device),
            edit_triple=Triple(s=s, p=p, o=o_new),
            original_object=o_old,
            n_probes=len(probes),
            butterfly_index=float(bi),
            pre_accuracy=float(pre_acc),
            post_accuracy=float(post_acc),
            details=details,
        )

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        return report


# =========================================================
# Example usage (capitals)
# =========================================================

if __name__ == "__main__":
    # 1) Provide an offline KG file (JSONL).
    # Create a tiny KG containing at least:
    # - (Canada, capital, Ottawa) plus other 1-hop facts about Canada and Ottawa
    kg_path = os.path.join(BASE_DIR, "mini_kg_capitals.jsonl")
    kg = LocalJsonlKG(kg_path)

    # 2) Provide relation metadata for induction.
    # For a minimal start, you can leave most empty.
    # Example ids are arbitrary; must match predicate ids in your KG.
    rel_meta = {
        "R:capital": RelationMeta(relation_id="R:capital", inverse_of=None, symmetric=False, transitive=False),
        "R:located_in": RelationMeta(relation_id="R:located_in", inverse_of="R:contains", symmetric=False, transitive=True),
        "R:contains": RelationMeta(relation_id="R:contains", inverse_of="R:located_in", symmetric=False, transitive=True),
    }

    runner = ButterflyKERunner(
        kg=kg,
        rel_meta=rel_meta,
        model_name="gpt2-xl",
    )

    # Example edit: Canada capital -> Toronto
    report = runner.run(
        s="E:Canada",
        p="R:capital",
        o_new="E:Toronto",
        method="memit",  # or "rome"
        neighbor_limit=200,
        max_probes=300,
        include_induction=True,
        out_path="results/butterflyke_canada_toronto.json",
    )

    print("Butterfly Index:", report.butterfly_index)
    print("Pre-acc:", report.pre_accuracy, "Post-acc:", report.post_accuracy)
    print("Num probes:", report.n_probes)
