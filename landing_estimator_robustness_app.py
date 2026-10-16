"""
Robustness comparison: alpha=0.25 (baseline) vs alpha=0.7 under varying noise/dropout.
Same scenarios/seeds; no controller/physics change.
"""
import argparse
from evaluation.estimator_robustness import main, SEEDS, ROBUST_CONFIGS

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="output/estimator_robustness")
    p.add_argument("--show", action="store_true")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    return p

def run(args=None):
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
    return main(outdir=args.outdir, show=args.show, seeds=seeds)

if __name__ == "__main__":
    run(build_parser().parse_args())
