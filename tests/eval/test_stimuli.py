"""Unit tests for jlensvl.eval.stimuli -- schema round-trip + validation.
No model, no torch: pure dataclass/JSON logic."""
import json

import pytest

from jlensvl.eval.stimuli import Concept, Stimulus, StimulusSet


def _tiny_set():
    concepts = {
        "paris": Concept("paris", ["Paris"]),
        "london": Concept("london", ["London"]),
        "berlin": Concept("berlin", ["Berlin"]),
    }
    items = [
        Stimulus("cap-fr", "The capital of France is", "paris", ["london", "berlin"], category="capital"),
        Stimulus("cap-uk", "The capital of the UK is", "london", ["paris", "berlin"], category="capital"),
    ]
    return StimulusSet(name="tiny", concepts=concepts, items=items, category="test")


def test_valid_set_constructs():
    s = _tiny_set()
    assert len(s.items) == 2
    assert len(s.concepts) == 3


def test_validate_rejects_undefined_target():
    concepts = {"paris": Concept("paris", ["Paris"])}
    items = [Stimulus("bad", "prompt", "atlantis", ["paris"])]
    with pytest.raises(ValueError, match="atlantis"):
        StimulusSet(name="bad-set", concepts=concepts, items=items)


def test_validate_rejects_undefined_distractor():
    concepts = {"paris": Concept("paris", ["Paris"])}
    items = [Stimulus("bad", "prompt", "paris", ["atlantis"])]
    with pytest.raises(ValueError, match="atlantis"):
        StimulusSet(name="bad-set", concepts=concepts, items=items)


def test_to_jsonl_writes_header_then_items(tmp_path):
    s = _tiny_set()
    path = tmp_path / "tiny.jsonl"
    s.to_jsonl(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1 + len(s.items)
    header = json.loads(lines[0])
    assert header["_header"] is True
    assert header["name"] == "tiny"
    assert header["category"] == "test"
    assert set(header["concepts"]) == {"paris", "london", "berlin"}
    first_item = json.loads(lines[1])
    assert first_item["id"] == "cap-fr"
    assert first_item["target"] == "paris"


def test_round_trip_preserves_content(tmp_path):
    s = _tiny_set()
    path = tmp_path / "tiny.jsonl"
    s.to_jsonl(path)
    reloaded = StimulusSet.from_jsonl(path)

    assert reloaded.name == s.name
    assert reloaded.category == s.category
    assert set(reloaded.concepts) == set(s.concepts)
    for name, concept in s.concepts.items():
        assert reloaded.concepts[name].words == concept.words

    assert len(reloaded.items) == len(s.items)
    for orig, back in zip(s.items, reloaded.items):
        assert orig.id == back.id
        assert orig.prompt == back.prompt
        assert orig.target == back.target
        assert orig.distractors == back.distractors
        assert orig.category == back.category
        assert orig.image == back.image
        assert orig.meta == back.meta


def test_from_jsonl_rejects_missing_header(tmp_path):
    path = tmp_path / "no_header.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt": "p", "target": "a", "distractors": []}) + "\n")
    with pytest.raises(ValueError, match="header"):
        StimulusSet.from_jsonl(path)


def test_from_jsonl_rejects_undefined_concept_reference(tmp_path):
    path = tmp_path / "bad_ref.jsonl"
    header = {"_header": True, "name": "bad", "category": "", "concepts": {"paris": ["Paris"]}}
    item = {"id": "x", "prompt": "p", "target": "paris", "distractors": ["atlantis"]}
    path.write_text(json.dumps(header) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="atlantis"):
        StimulusSet.from_jsonl(path)


def test_from_jsonl_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        StimulusSet.from_jsonl(path)


def test_stimulus_defaults():
    s = Stimulus("x", "prompt", "a", ["b"])
    assert s.image is None
    assert s.category == ""
    assert s.meta == {}


def test_seed_sets_load_and_validate():
    """The two hand-authored seed sets must load and pass schema validation."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for fname in ("association_text.jsonl", "multilingual_text.jsonl"):
        path = repo_root / "data" / "eval_sets" / fname
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")
        s = StimulusSet.from_jsonl(path)
        assert len(s.items) >= 10
        assert len(s.concepts) >= 1
