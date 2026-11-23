"""CLI for the long-running Phase II jobs: ``python -m phase2 attributions|representations``."""

import sys

import pipeline as P

if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "attributions"
    sample = P.load_sample()
    if job == "attributions":
        P.run_attributions(sample)
    elif job == "representations":
        from . import representations as R

        R.extract(sample, force=True)
    else:
        raise SystemExit(f"unknown job {job}")
