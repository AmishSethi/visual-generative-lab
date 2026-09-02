#!/usr/bin/env python
"""Audit compositional datasets for the every-value-seen constraint.

A held-out combination only tests compositional generalization if each of its values appears,
in some other combination, in the training set. This compares whole values (a position is an
(x, y) pair, not two axes) against the dataset's own value lists in dataset_metadata.json:
in-range values are property_ranges minus the values that occur only in the OOD splits; every
in-range value must occur in train/.

Exit status is the number of violating datasets, so it can gate a pipeline.

Usage: python -m paper_runs.table3.audit_compositional DATASET_ROOT [--compare OTHER_ROOT]
"""
import argparse, json, pathlib, re, sys

NUM = re.compile(r"^(r|x|y|rot|cnt)(n?)(\d+)(?:p(\d+))?$")

def parse_num(tok):
    m = NUM.match(tok)
    if not m: return None
    key, neg, whole, frac = m.groups()
    v = float(f"{whole}.{frac or 0}") * (-1 if neg else 1)
    return key, (int(v) if key == "cnt" else v)

def folder_values(name, props):
    """Map a class-folder name to {property: value} for the given property order."""
    toks = name.split("_"); nums = {}; cats = []
    for t in toks:
        p = parse_num(t)
        if p: nums[p[0]] = p[1]
        else: cats.append(t)
    out = {}
    for prop in props:
        if prop == "radius": out[prop] = nums.get("r")
        elif prop == "position": out[prop] = (nums.get("x"), nums.get("y"))
        elif prop == "rotation": out[prop] = nums.get("rot")
        elif prop == "count": out[prop] = nums.get("cnt")
        elif prop in ("shape", "color"):
            out[prop] = next((c for c in cats if c not in out.values()), None)
            if out[prop] is not None: cats.remove(out[prop])
    return out

def canon(prop, v):
    if prop == "position": return (float(v[0]), float(v[1]))
    if prop == "color": return v[0] if isinstance(v, list) else v
    if prop == "count": return int(v)
    return float(v) if isinstance(v, (int, float)) else v

def split_values(d, split, props):
    vals = {p: set() for p in props}
    if (d / split).is_dir():
        for f in (d / split).iterdir():
            if f.is_dir():
                fv = folder_values(f.name, props)
                for p in props:
                    if fv.get(p) is not None: vals[p].add(canon(p, fv[p]))
    return vals

def folders(d):
    return {sp: sorted(p.name for p in (d / sp).iterdir() if p.is_dir())
            for sp in ("train", "test_compositional", "test_prop_a_ood", "test_prop_b_ood", "test_both_ood") if (d / sp).is_dir()}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path); ap.add_argument("--compare", type=pathlib.Path, default=None)
    a = ap.parse_args()
    dsets = sorted(p for p in a.root.iterdir() if p.is_dir() and (p / "train").is_dir())
    viol = 0
    print(f"{len(dsets)} datasets under {a.root}")
    for d in dsets:
        meta = json.loads((d / "dataset_metadata.json").read_text()); props = meta["include_properties"]
        full = {p: {canon(p, v) for v in meta["property_ranges"][p]} for p in props}
        train = split_values(d, "train", props)
        ood = {p: set() for p in props}
        for sp in ("test_prop_a_ood", "test_prop_b_ood", "test_both_ood"):
            for p, s in split_values(d, sp, props).items(): ood[p] |= s
        inrange = {p: full[p] - (ood[p] - train[p] - split_values(d, "test_compositional", props)[p]) for p in props}
        missing = {p: sorted(map(str, inrange[p] - train[p])) for p in props if inrange[p] - train[p]}
        line = f"  {d.name:26s} train combos={len(folders(d)['train']):4d}  " + ("VIOLATION " + str(missing) if missing else "ok")
        if a.compare is not None and (a.compare / d.name).is_dir():
            line += "   | vs reference: " + ("identical" if folders(d) == folders(a.compare / d.name) else "DIFFERENT combination sets")
        print(line); viol += bool(missing)
    print(f"VIOLATIONS: {viol}/{len(dsets)}"); sys.exit(viol)

if __name__ == "__main__":
    main()
