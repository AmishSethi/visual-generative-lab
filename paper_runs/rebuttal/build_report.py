#!/usr/bin/env python
"""Collect every rebuttal result into one markdown summary.

Reads whatever JSON files exist under REBUTTAL/eval and renders the tables the
rebuttal needs, marking anything still missing as PENDING so the gaps are
visible rather than silently absent.
"""
import argparse
import json
from pathlib import Path

from paper_runs.rebuttal.manifest import EVAL_ROOT, REBUTTAL_ROOT


def load(name):
    path = EVAL_ROOT / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_merged(name, pattern):
    """Load an aggregate file, or rebuild it from per-member shards.

    The long serial evaluations were split into short parallel jobs so they would
    backfill onto the cluster in time, and each shard writes its own file rather
    than the single aggregate this report was written against.  The aggregate
    still wins if a whole-family job did run.

    Merging is recursive because the shards nest: a scaling shard is
    {"position": {"10k": ...}} and a realistic shard is {"skills": {"size": ...}},
    so a flat update would let each shard's top-level key erase the previous one.
    """
    def deep_merge(into, other):
        for key, value in other.items():
            if isinstance(value, dict) and isinstance(into.get(key), dict):
                deep_merge(into[key], value)
            else:
                into[key] = value

    # The aggregate is merged as the base rather than short-circuiting on it: a
    # whole-family job may have written a partial file (realistic_results.json
    # holds only the ground-truth ceilings from a --validate-only pass), and
    # returning that early would silently hide every per-skill shard.
    merged = {}
    whole = load(name)
    if isinstance(whole, dict):
        deep_merge(merged, whole)
    for path in sorted(EVAL_ROOT.glob(pattern)):
        if path.name == name:
            continue
        part = load(path.name)
        if isinstance(part, dict):
            deep_merge(merged, part)
    return merged or None


def section(title, reviewer, body):
    return f"\n## {title}\n*{reviewer}*\n\n{body}\n"


def fmt_matched_ood(data):
    if not data:
        return "PENDING"
    skills = list(data["skills"])
    lines = [f"delta = distance outside training support / training span. "
             f"Spans: " + ", ".join(f"{k} {v:g}" for k, v in data["spans"].items()) + ".", ""]
    lines.append("| delta bin | " + " | ".join(skills) + " |")
    lines.append("|---" * (len(skills) + 1) + "|")
    bins = sorted({b for s in skills for b in data["skills"][s]["bins"]})
    for b in bins:
        row = [b]
        for s in skills:
            entry = data["skills"][s]["bins"].get(b)
            row.append(f"{entry['mean']:.1f} ± {entry['std']:.1f}" if entry else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Largest normalised distance each skill is actually evaluated at: " + ", ".join(
        f"**{s} {data['skills'][s]['max_delta_evaluated']:.3f}** ({data['skills'][s]['num_seeds']} seeds)"
        for s in skills))
    return "\n".join(lines)


def fmt_vae(data):
    if not data:
        return "PENDING"
    lines = [f"VAE: `{data['vae']}` (the one the paper's latent runs use). "
             "Ground-truth renders encoded then decoded, scored with the paper's metrics.", "",
             "| resolution | size mean IoU | % radii ≥ 0.90 | position err (px) | rotation err (deg) |",
             "|---|---|---|---|---|"]
    for res, entry in data["resolutions"].items():
        lines.append(
            f"| {res}×{res} | {entry['size']['mean_iou']:.3f} | "
            f"{entry['size']['pct_radii_above_0.90_iou']:.0f}% | "
            f"{entry['position']['mean_position_error_px']:.2f} | "
            f"{entry['rotation']['mean_angle_error_deg']:.2f} |")
    return "\n".join(lines)


def fmt_counters(data):
    if not data:
        return "PENDING"
    lines = ["Counter accuracy on **ground-truth** renders (upper bound on any generator's score):", "",
             "| placement | watershed (paper's) | Hough | LoG |", "|---|---|---|---|"]
    for label in ("non_overlapping", "random_placement"):
        key = f"{label}_train_range_mean"
        if key in data:
            m = data[key]
            lines.append(f"| {label.replace('_', ' ')} (N=2..7) | "
                         f"{m['watershed']:.1f}% | {m['hough']:.1f}% | {m['log']:.1f}% |")
    if "non_overlapping" in data:
        lines += ["", "Per-count, non-overlapping:", "",
                  "| N | watershed | Hough | LoG |", "|---|---|---|---|"]
        for n in sorted(data["non_overlapping"]["watershed"], key=lambda x: int(x)):
            lines.append(f"| {n} | {data['non_overlapping']['watershed'][n]:.0f}% | "
                         f"{data['non_overlapping']['hough'][n]:.0f}% | "
                         f"{data['non_overlapping']['log'][n]:.0f}% |")
    return "\n".join(lines)


def fmt_count_samples(data):
    """Model-side half of yLdq q3: the same generations under three counters."""
    if not data or "split_means" not in data:
        return ""
    lines = ["", "Re-scoring the model's own generations (3 seeds) with each counter:", "",
             "| split | watershed (paper's) | Hough | LoG |", "|---|---|---|---|"]
    for split, means in data["split_means"].items():
        lines.append(f"| {split} | {means['watershed']:.1f}% | {means['hough']:.1f}% | "
                     f"{means['log']:.1f}% |")
    return "\n".join(lines)


def fmt_count_overlap(data):
    """Does merging explain the count failure, or does the model miscount?

    Every disc has the same area, so total foreground ink divided by that unit
    recovers the count even when discs merge into a single blob -- an estimator
    that overlap cannot fool, and which is exact on ground truth.  Comparing it
    against the paper's watershed separates measurement error from model error.
    """
    if not data:
        return ""
    from collections import defaultdict
    by = defaultdict(list)
    for r in data:
        by[r["want"]].append(r)
    pct = lambda g, f: 100.0 * sum(1 for r in g if f(r)) / len(g)
    lines = ["", "Merging audit over every regenerated sample "
                 f"(n = {len(data)}), overlap-robust counter vs the paper's watershed:", "",
             "| N | merged pair present | watershed ±1 | overlap-robust ±1 |",
             "|---|---|---|---|"]
    for N in sorted(by):
        g = by[N]
        lines.append(f"| {N} | {pct(g, lambda r: r['cc'] < r['area']):.0f}% | "
                     f"{pct(g, lambda r: abs(r['ws']-N) <= 1):.0f}% | "
                     f"{pct(g, lambda r: abs(r['area']-N) <= 1):.0f}% |")
    wrong = [r for r in data if r["ws"] != r["want"]]
    rescued = sum(1 for r in wrong if r["area"] == r["want"])
    lines += ["", f"Of the {len(wrong)} generations the watershed scores wrong, the "
                  f"overlap-robust counter rescues {rescued} "
                  f"({100.0*rescued/len(wrong):.1f}%); the remainder are genuine miscounts."]
    return "\n".join(lines)


def fmt_shape_vocab(data):
    if not data or not data.get("cells"):
        return "PENDING"
    cells = sorted(data["cells"], key=lambda c: (c["S"], c["C"], c["coverage"]))
    lines = ["Fixed 15,000-image budget in every cell. **B** = held-out combinations "
             "(compositional interpolation), mean ± std over seeds.", "",
             "| run | S | C | coverage | K | samples/K | n | B shape | B colour | B joint |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['name']} | {c['S']} | {c['C']} | {c['coverage']:.0%} | {c['K']} | "
                     f"{c['samples_per_combo']} | {c['n_seeds']} | {c['shape']:.1f}% | "
                     f"{c['color']:.1f}% | {c['joint']:.1f} ± {c['joint_std']:.1f} |")
    reg = data.get("regression")
    if reg:
        lines += ["", f"Regression on {reg['n_cells']} cells: accuracy = {reg['intercept']:.1f} "
                      f"+ {reg['beta_coverage']:.1f}·coverage + {reg['beta_log2K']:.1f}·log2(K). "
                      f"R² full {reg['r2_full']:.3f}; unique to coverage "
                      f"{reg['unique_to_coverage']:.3f}, unique to log2(K) "
                      f"{reg['unique_to_logK']:.3f}."]
    by_name = {c["name"]: c for c in cells}
    if "s4_cov50" in by_name and "s8_cov25" in by_name:
        a, b = by_name["s4_cov50"]["joint"], by_name["s8_cov25"]["joint"]
        lines += ["", f"**Decisive pair** (identical K=16 and identical samples/combination, "
                      f"coverage 50% vs 25%): S=4@50% = {a:.1f}%, S=8@25% = {b:.1f}%, "
                      f"difference {a - b:+.1f} pp."]
    return "\n".join(lines)


def fmt_scaling(data):
    if not data:
        return "PENDING"
    lines = ["Compute held fixed at 78,125 gradient steps; only the number of unique "
             "training images changes. Nested datasets (10k ⊂ 20k ⊂ 40k ⊂ 80k).", "",
             "| skill | dataset | train | interp | extra |", "|---|---|---|---|---|"]
    for skill, by_size in data.items():
        for tag, res in by_size.items():
            s = res["splits"]
            lines.append(f"| {skill} | {tag} | {s['train']['accuracy']:.1f}% | "
                         f"{s['interp']['accuracy']:.1f}% | {s['extra']['accuracy']:.1f}% |")
    return "\n".join(lines)


def fmt_hires(data, label):
    if not data:
        return "PENDING"
    t = data.get("thresholds", {})
    lines = [f"Thresholds scaled to mean the same thing as at 64×64: "
             f"IoU ≥ {t.get('size_iou', 0.9)}, position ≤ {t.get('position_px', 4)} px, "
             f"rotation ≤ {t.get('rotation_deg', 5)}°.", "",
             "| skill | train | interp | extra |", "|---|---|---|---|"]
    for skill, res in data.get("skills", {}).items():
        s = res["splits"]
        lines.append(f"| {skill} | {s.get('train', float('nan')):.1f}% | "
                     f"{s.get('interp', float('nan')):.1f}% | {s.get('extra', float('nan')):.1f}% |")
    return "\n".join(lines)


def fmt_t2i(data):
    if not data:
        return "PENDING"
    models = [k for k in data if k != "validation_on_vgl_ground_truth"]
    if not models:
        return "PENDING"
    lines = []
    val = data.get("validation_on_vgl_ground_truth")
    if val:
        lines += ["Scorer validated on VGL ground truth first: " + ", ".join(
            f"{k} {v:.1f}" for k, v in val.items()), ""]
    lines += ["`on-protocol` = accuracy over the images where the model actually produced "
              "a flat subject on a plain light background; `compliance` = how often it did.", "",
              "| model | skill | raw acc | on-protocol acc | compliance |",
              "|---|---|---|---|---|"]
    for model in models:
        for skill, res in data[model].items():
            given = res.get("accuracy_given_compliant")
            given_str = "n/a" if given is None else f"{given:.1f}%"
            lines.append(f"| {model} | {skill} | {res['overall_accuracy']:.1f}% | "
                         f"{given_str} | {res['compliance_rate']:.0f}% |")
    return "\n".join(lines)


def fmt_masks(data):
    if not data:
        return "PENDING"
    import statistics
    from collections import defaultdict

    groups = defaultdict(list)
    for name, res in data.items():
        cov = name.split("_cov")[1].split("_mask")[0]
        groups[cov].append((name.split("mask")[-1], res["categories"]["B_heldout"]))

    lines = ["Same S=4 vocabulary and coverage, five different random combination masks. "
             "Reported at two coverages because 50% is saturated and cannot show spread.", ""]
    for cov in sorted(groups, key=int, reverse=True):
        rows = sorted(groups[cov], key=lambda r: int(r[0]))
        vals = [b["joint_accuracy"] for _, b in rows]
        lines += [f"**{cov}% coverage**", "",
                  "| mask seed | B shape | B colour | B joint |", "|---|---|---|---|"]
        for seed, b in rows:
            lines.append(f"| {seed} | {b['shape_accuracy']:.1f}% | {b['color_accuracy']:.1f}% | "
                         f"{b['joint_accuracy']:.1f}% |")
        spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines += ["", f"mean {statistics.mean(vals):.1f}% ± {spread:.1f} "
                      f"(range {min(vals):.1f}–{max(vals):.1f})", ""]
    return "\n".join(lines)


def fmt_memorization(data):
    if not data:
        return "PENDING"
    lines = ["Distance from each generation to its nearest training image, against the same "
             "quantity for fresh held-out ground-truth renders.", "",
             "| skill | space | gen NN dist | held-out NN dist | ratio | gen copy rate |",
             "|---|---|---|---|---|---|"]
    for skill, spaces in data.items():
        for space, res in spaces.items():
            g, h = res["generations"], res["held_out_real"]
            lines.append(f"| {skill} | {space} | {g['mean_nn_distance']:.4f} | "
                         f"{h['mean_nn_distance']:.4f} | {res['gen_over_heldout_nn_ratio']:.3f} | "
                         f"{g['copy_rate_pct']:.1f}% |")
    return "\n".join(lines)



def fmt_ranges(data, analysis):
    if not data:
        return "PENDING"
    lines = ["Same architecture and budget; only the training support moves.", "",
             "| run | skill | support | train | interp | extra |", "|---|---|---|---|---|---|"]
    for name, res in data.items():
        s = res["splits"]
        lines.append(f"| {name} | {res['skill']} | — | {s.get('train', 0):.1f}% | "
                     f"{s.get('interp', 0):.1f}% | {s.get('extra', 0):.1f}% |")
    if analysis:
        lines += ["", "Break point (furthest query beyond the boundary still above 50%):", "",
                  "| run | span | break beyond (px) | break beyond (÷span) |", "|---|---|---|---|"]
        for r in analysis:
            lines.append(f"| {r['run']} | {r['span']:.0f} | {r['break_above_px']:.1f} | "
                         f"{r['break_above_norm']:.3f} |")
    return "\n".join(lines)


def fmt_text_cond(data):
    if not data:
        return "PENDING"
    lines = ["Skill value supplied as a frozen CLIP embedding of a sentence describing it, "
             "instead of as a scalar. Same datasets and schedule as the paper's baseline.", "",
             "| skill | train | interp | extra |", "|---|---|---|---|"]
    for name, res in data.items():
        s = res["splits"]
        lines.append(f"| {name} | {s.get('train', 0):.1f}% | {s.get('interp', 0):.1f}% | "
                     f"{s.get('extra', 0):.1f}% |")
    return "\n".join(lines)


def fmt_realistic(data):
    if not data:
        return "PENDING"
    lines = []
    ceilings = data.get("metric_ceilings_on_ground_truth")
    if ceilings:
        lines += ["Metric ceilings on ground-truth complex renders (all 100% before use): " +
                  ", ".join(f"{k.split('_')[0]} {v:.0f}" for k, v in ceilings.items()
                            if k.endswith(("threshold", "accuracy"))), ""]
    lines += ["| skill | train | interp | extra |", "|---|---|---|---|"]
    for skill, res in data.get("skills", {}).items():
        s = res["splits"]
        lines.append(f"| {skill} | {s.get('train', {}).get('accuracy', 0):.1f}% | "
                     f"{s.get('interp', {}).get('accuracy', 0):.1f}% | "
                     f"{s.get('extra', {}).get('accuracy', 0):.1f}% |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build the rebuttal results summary.")
    parser.add_argument("--out", type=Path, default=REBUTTAL_ROOT / "REBUTTAL_RESULTS.md")
    args = parser.parse_args()

    parts = ["# VGL rebuttal — measured results",
             "",
             "All numbers produced by `paper_runs/rebuttal/`. Anything marked PENDING is "
             "still running.",
             ""]

    parts.append(section(
        "1. Pretrained text-to-image models under VGL metrics",
        "Reviewer HeVs w5 / AC key concern: does the toy setting say anything about real models?",
        fmt_t2i(load("t2i/scores.json"))))

    parts.append(section(
        "2. Shape vocabulary: coverage fraction vs number of combinations",
        "Reviewer XDWw / AC-required",
        fmt_shape_vocab(load("coverage_vs_k_analysis.json"))))

    parts.append(section(
        "3. Does more data fix extrapolation?",
        "Reviewer HeVs w3",
        fmt_scaling(load_merged("scaling_results.json", "scaling_*.json"))))

    parts.append(section(
        "4. 128×128 replication",
        "Reviewers HeVs w2/w5, yLdq w1",
        fmt_hires(load("hires128_results.json"), "pixel")))

    parts.append(section(
        "5. VAE reconstruction ceiling",
        "Reviewer HeVs w2: is the weak latent result a low-resolution artifact?",
        fmt_vae(load("vae_ceiling.json"))))

    parts.append(section(
        "6. Extrapolation at matched normalised OOD distance",
        "Reviewer yLdq q1",
        fmt_matched_ood(load("matched_ood_distance.json"))))

    parts.append(section(
        "7. Is the count failure the model or the evaluator?",
        "Reviewer yLdq q3",
        fmt_counters(load("counter_crossval.json"))
        + fmt_count_samples(load("count_samples_scored.json"))
        + fmt_count_overlap(load("count_overlap_audit.json"))))

    parts.append(section(
        "8. Variance across random coverage masks",
        "Reviewer yLdq q2",
        fmt_masks(load("shape_vocab_masks_results_seed0.json"))))

    parts.append(section(
        "9. Training-range variation: does the break point follow the support?",
        "Reviewer yLdq w2",
        fmt_ranges(load_merged("ranges_results.json", "ranges_*.json"), load("ranges_analysis.json"))))

    parts.append(section(
        "10. Text-encoder conditioning instead of numeric",
        "Reviewer HeVs w1/w6",
        fmt_text_cond(load_merged("text_cond_results.json", "text_cond_*.json"))))

    parts.append(section(
        "11. Visually complex renders",
        "Reviewer HeVs w5",
        fmt_realistic(load_merged("realistic_results.json", "realistic_*.json"))))

    parts.append(section(
        "12. Memorisation probe",
        "Reviewer HeVs w3",
        fmt_memorization(load("memorization_probe.json"))))

    text = "\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\n\nwrote {args.out}")


if __name__ == "__main__":
    main()
