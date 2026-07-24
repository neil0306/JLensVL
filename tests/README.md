# JLensVL tests

Run from the repo root with the offline HF env and both `src/` (JLensVL) and
the sibling `jacobian-lens` engine on `PYTHONPATH` (also auto-added by
`conftest.py` as a fallback):

```
HF_HOME=/home/anu/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=$PWD/src:/home/anu/src/ning/J-space-test/jacobian-lens \
~/miniconda3/envs/jlensvl/bin/python -m pytest tests/ -q
```

`test_lens_math.py` and `test_viz.py` are pure CPU/data-in-data-out tests and
always run. `test_tokenization.py` needs the cached Qwen3.5-4B tokenizer and
skips its module if that isn't loadable offline. `test_integration.py`
requires the full Qwen3.5-4B model weights (not yet downloaded) and is
gated behind `JLENSVL_HAVE_WEIGHTS=1` + `CUDA_VISIBLE_DEVICES=1`; it
auto-skips until then.
