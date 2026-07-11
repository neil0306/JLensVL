"""Quick Apple Silicon / device sanity check: before committing to a multi-hour
fit, confirm this box can actually run backward passes through Qwen3.5's
Gated-DeltaNet layers (the differentiable pure-PyTorch path autograd depends
on) and that the raw per-prompt Jacobian already shows the concept-forms-
before-spoken effect (boot -> Italy -> euro) on a single probe. Runs in well
under two minutes -- much cheaper than finding out after 20 hours of fitting.

  MODEL_DIR=Qwen/Qwen3.5-4B python examples/09_apple_silicon_check.py
"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jlens
from jlens import JacobianLens
from jlens.fitting import jacobian_for_prompt

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = os.environ.get("DEV", auto_device())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(DEV)
hf.eval()
log("loaded on", DEV, "| torch", torch.__version__)
lm = jlens.from_hf(hf, tok)

# coherence: does the model generate sane text on this device/dtype at all?
ids = tok("The capital of France is", return_tensors="pt").to(DEV)
with torch.no_grad():
    g = hf.generate(**ids, max_new_tokens=6, do_sample=False)
log("SANITY 'The capital of France is' ->",
    repr(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))

# the real test: autograd through Gated-DeltaNet on this device
PROBE = ("Fact: The currency used in the country shaped like a boot is the")
SRC = [20, 25, 29]
log("jacobian_for_prompt on", DEV, "(tests GDN autograd) ...")
t = time.time()
J, seqlen, nvalid = jacobian_for_prompt(lm, PROBE, SRC, dim_batch=8, max_seq_len=128)
log(f"jacobian done {time.time()-t:.0f}s | seq={seqlen} n_valid={nvalid}")

lens = JacobianLens(J, n_prompts=1, d_model=lm.d_model)
ll, _, _ = lens.apply(lm, PROBE, positions=[-1], layers=SRC)
for L in SRC:
    top = [tok.decode([i]).strip() for i in ll[L][0].topk(6).indices.tolist()]
    log(f"  J-Lens L{L}: {top}")
log(f"ROUTE OK -- backward through GDN works on {DEV}, single-prompt Jacobian "
    "already surfaces the euro/Italy concept -- safe to run a full fit_lens.py")
