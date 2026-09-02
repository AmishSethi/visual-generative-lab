#!/usr/bin/env python
"""Audit compositional datasets for the every-value-seen constraint.

A held-out combination only tests compositional generalization if each of its values
appears, in some other combination, in the training set. Otherwise the model is asked
about an unseen value, not an unseen pairing. This checks every dataset under a root:
for each property, the set of in-range values (those appearing in train or
test_compositional) must be a subset of the values appearing in train.

Exit status is the number of violating datasets, so it can gate a pipeline.

Usage:  python audit_compositional.py DATASET_ROOT [--compare OTHER_ROOT]
"""
import collections, pathlib, re, sys

def parse_dir(name):
    out = {}
    for tok in name.split("_"):
        m = re.match(r"^(r|p|x|y|rot|cnt|ang)(-?\d+(?:\.\d+)?)$", tok)
        if m: out[m.group(1)] = m.group(2)
        elif re.match(r"^[a-z]+$", tok): out.setdefault("cat", []).append(tok)
    return out

def values(d, splits):
    v = collections.defaultdict(set)
    for sp in splits:
        if (d / sp).is_dir():
            for p in (d / sp).iterdir():
                if p.is_dir():
                    for k, x in parse_dir(p.name).items():
                        if k == "cat":
                            for j, cv in enumerate(x): v[f"cat{j}"].add(cv)
                        else: v[k].add(x)
    return v

def folders(d):
    return {sp: sorted(p.name for p in (d / sp).iterdir() if p.is_dir())
            for sp in ("train", "test_compositional", "test_prop_a_ood", "test_prop_b_ood", "test_both_ood") if (d / sp).is_dir()}

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path, help="dataset root to audit")
    ap.add_argument("--compare", type=pathlib.Path, default=None, help="reference root; report whether combination sets match")
    a = ap.parse_args(); root, other = a.root, a.compare
    dsets = sorted(p for p in root.iterdir() if p.is_dir() and (p / "train").is_dir())
    viol = 0
    print(f"{len(dsets)} datasets under {root}")
    for d in dsets:
        seen = values(d, ["train"]); inrange = values(d, ["train", "test_compositional"])
        missing = {k: sorted(inrange[k] - seen[k]) for k in inrange if inrange[k] - seen[k]}
        line = f"  {d.name:26s} train combos={sum(1 for p in (d/'train').iterdir() if p.is_dir()):4d}  " + ("VIOLATION " + str(missing) if missing else "ok")
        if other is not None and (other / d.name).is_dir():
            same = folders(d) == folders(other / d.name)
            line += "   | vs reference: " + ("identical combination sets" if same else "DIFFERENT combination sets")
        print(line); viol += bool(missing)
    print(f"VIOLATIONS: {viol}/{len(dsets)}")
    sys.exit(viol)

if __name__ == "__main__":
    main()
