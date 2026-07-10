"""Export a torch-fitted Jacobian lens (.pt) to a plain .npz so MLX (or numpy)
can load the per-layer J_ell matrices. Run in the torch venv (needs jlens).

  python scripts/lens_to_npz.py lens.pt lens_jl.npz
"""
import sys
import numpy as np
from jlens import JacobianLens

src = sys.argv[1] if len(sys.argv) > 1 else "lens.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else "lens_jl.npz"
lens = JacobianLens.from_pretrained(src)
out = {str(l): J.float().cpu().numpy().astype(np.float16) for l, J in lens.jacobians.items()}
np.savez(dst, **out)
print(f"saved {dst} | layers {sorted(lens.jacobians.keys())} | d_model {lens.d_model}")
