"""Controlled-stimuli / concept-vocabulary schema for JLensVL quantitative eval.

A `StimulusSet` is a small labeled dataset: a shared vocabulary of `Concept`s
(surface word forms) plus a list of `Stimulus` items, each naming one
`target` concept and one or more `distractors` concepts by name, resolved
against the set's `concepts` dict. It round-trips through JSONL: `to_jsonl`
writes one header line (set metadata + the concept vocabulary) followed by
one line per item; `from_jsonl` reads it back. Kept deliberately simple
(stdlib `json` only, no schema library) so seed data stays auditable by eye
and diffable in review.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Concept:
    """A concept and its surface word forms.

    `words` feeds `JLensVL._word_ids`-style scoring: each word is tried bare,
    leading-space, capitalized, and leading-space-capitalized, keeping
    whichever variant tokenizes to a single vocab id. Multiple words (e.g.
    translations, synonyms) let one concept pool evidence across surface
    forms -- e.g. a multilingual answer.
    """

    name: str
    words: list[str]


@dataclass
class Stimulus:
    """One labeled eval item.

    `target` and every entry of `distractors` must be keys into the owning
    `StimulusSet.concepts` -- `StimulusSet` validates this at construction.
    `image` is an optional path; leave it `None` for text-only items (scored
    via the text-scoring path instead of `concept_race`).
    """

    id: str
    prompt: str
    target: str
    distractors: list[str]
    image: str | None = None
    category: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class StimulusSet:
    """A named collection of `items` plus the `concepts` vocabulary they draw on."""

    name: str
    concepts: dict[str, Concept]
    items: list[Stimulus]
    category: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if any item references an undefined concept."""
        errors = []
        for item in self.items:
            missing = [n for n in (item.target, *item.distractors) if n not in self.concepts]
            if missing:
                errors.append(f"item {item.id!r} references undefined concept(s): {missing}")
        if errors:
            raise ValueError("; ".join(errors))

    # ---------- JSONL round-trip ----------
    def to_jsonl(self, path) -> None:
        """Write a header line (`{'_header': True, name, category, concepts}`)
        followed by one JSON object per item, in `self.items` order."""
        header = {
            "_header": True,
            "name": self.name,
            "category": self.category,
            "concepts": {n: c.words for n, c in self.concepts.items()},
        }
        lines = [json.dumps(header, ensure_ascii=False)]
        lines += [json.dumps(asdict(item), ensure_ascii=False) for item in self.items]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path) -> "StimulusSet":
        """Read back a file written by `to_jsonl`.

        Raises ValueError if the file is empty, the first line isn't a
        header, or (via `validate`) an item references an undefined concept.
        """
        raw_lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw_lines:
            raise ValueError(f"{path}: empty stimulus set file")
        header = json.loads(raw_lines[0])
        if not header.get("_header"):
            raise ValueError(f"{path}: first line is not a StimulusSet header (missing '_header': true)")
        concepts = {n: Concept(name=n, words=list(words)) for n, words in header["concepts"].items()}
        items = []
        for ln in raw_lines[1:]:
            d = json.loads(ln)
            items.append(
                Stimulus(
                    id=d["id"],
                    prompt=d["prompt"],
                    target=d["target"],
                    distractors=list(d["distractors"]),
                    image=d.get("image"),
                    category=d.get("category", ""),
                    meta=d.get("meta") or {},
                )
            )
        return cls(name=header["name"], concepts=concepts, items=items, category=header.get("category", ""))
