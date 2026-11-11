"""Same-seed repeats, G2 (Paper3 revision analysis addendum v1.0, Section 8): training records.

Each replication run is compared with its G2 repeat (same name, same seed):
  results.csv: number of epochs; whether every metric and loss column agrees epoch by epoch,
               ignoring the wall-clock column "time"; the first epoch at which any value
               differs, and the largest absolute difference per column;
  best.pt:     the selected epoch; the number of tensors in the saved model that are bitwise
               identical; the largest absolute difference over all floating-point tensors.
               File hashes are not compared (the checkpoint embeds its save date).
The comparison of cached test predictions is a separate step (it needs the G2 caches).
Needs torch and ultralytics (to read the checkpoints); CPU only.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

NAMES = ["baseline_seed42", "baseline_seed43", "p2_seed42", "p2_seed43"]
SKIP_ARGS = {"project", "name", "save_dir"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with open(path, newline="") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def compare_csv(a, b):
    cols = [c for c in a[0] if not c == "time"]
    out = {"epochs": [len(a), len(b)], "columns_compared": len(cols), "first_difference": None, "max_abs_diff": {}}
    for i, (x, y) in enumerate(zip(a, b)):
        d = [c for c in cols if not x.get(c) == y.get(c)]
        if d and out["first_difference"] is None:
            out["first_difference"] = {"epoch": i + 1, "columns": d,
                                       "replication": {c: x.get(c) for c in d[:6]}, "repeat": {c: y.get(c) for c in d[:6]}}
    for c in cols:
        m = 0.0
        for x, y in zip(a, b):
            try:
                m = max(m, abs(float(x[c]) - float(y[c])))
            except (TypeError, ValueError):
                pass
        out["max_abs_diff"][c] = m
    out["identical_except_time"] = out["first_difference"] is None and len(a) == len(b)
    return out


def compare_ckpt(pa, pb):
    import torch
    a, b = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
    sa, sb = a["model"].state_dict(), b["model"].state_dict()
    keys = sorted(set(sa) | set(sb))
    same, maxd, differ, missing = 0, 0.0, [], []
    for k in keys:
        if k not in sa or k not in sb:
            missing.append(k)
            continue
        x, y = sa[k], sb[k]
        if x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y):
            same += 1
            continue
        differ.append(k)
        if x.is_floating_point() and x.shape == y.shape:
            maxd = max(maxd, float((x.double() - y.double()).abs().max()))
    ta = {k: v for k, v in (a.get("train_args") or {}).items() if k not in SKIP_ARGS}
    tb = {k: v for k, v in (b.get("train_args") or {}).items() if k not in SKIP_ARGS}
    return {
        "selected_epoch": [a.get("epoch"), b.get("epoch")],
        "best_fitness": [a.get("best_fitness"), b.get("best_fitness")],
        "tensors": len(keys), "bitwise_identical": same, "differing": len(differ), "missing": missing,
        "first_differing_tensors": differ[:5], "max_abs_diff_float_tensors": maxd,
        "dtype": str(next(iter(sa.values())).dtype),
        "train_metrics_equal": a.get("train_metrics") == b.get("train_metrics"),
        "train_args_differences": {k: [ta.get(k), tb.get(k)] for k in sorted(set(ta) | set(tb)) if not ta.get(k) == tb.get(k)},
        "versions": [a.get("version"), b.get("version")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-root", required=True, help="Folder of the replication runs, e.g. $P3/runs")
    ap.add_argument("--rep-root", required=True, help="Folder of the G2 repeats, e.g. $P3/runs_g2")
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    for n in args.names:
        for root in (args.orig_root, args.rep_root):
            d = os.path.join(root, n)
            if not os.path.exists(os.path.join(d, "recovery_metadata.json")):
                sys.exit("ERROR: %s is not finished (recovery_metadata.json missing)" % d)
    res, inputs = {}, {}
    for n in args.names:
        da, db = os.path.join(args.orig_root, n), os.path.join(args.rep_root, n)
        ma, mb = json.load(open(os.path.join(da, "recovery_metadata.json"))), json.load(open(os.path.join(db, "recovery_metadata.json")))
        r = {"csv": compare_csv(rows(os.path.join(da, "results.csv")), rows(os.path.join(db, "results.csv"))),
             "best_pt": compare_ckpt(os.path.join(da, "weights", "best.pt"), os.path.join(db, "weights", "best.pt")),
             "hosts": [ma.get("hostname"), mb.get("hostname")]}
        res[n] = r
        for d in (da, db):
            for f in ("results.csv", os.path.join("weights", "best.pt")):
                inputs[os.path.join(d, f)] = sha256_file(os.path.join(d, f))
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = None
    rec = {"analysis": "G2 same-seed repeats, training records (addendum v1.0 Section 8)",
           "run_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": inputs,
           "versions": {"python": sys.version.split()[0], "torch": tv, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__), "pairs": res}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "g2_training.json"), "w") as f:
        json.dump(rec, f, indent=2)
    lines = ["== G2: replication run vs same-seed repeat (hosts: %s -> %s)" % tuple(res[args.names[0]]["hosts"])]
    for n in args.names:
        c, b = res[n]["csv"], res[n]["best_pt"]
        fd = c["first_difference"]
        lines.append("  %-16s results.csv: %d vs %d epochs; %s" % (
            n, c["epochs"][0], c["epochs"][1],
            "every value identical except time" if c["identical_except_time"] else
            ("first difference at epoch %d in %d columns, e.g. %s" % (fd["epoch"], len(fd["columns"]), fd["columns"][:3])
             if fd else "values identical over the shared epochs, epoch counts differ")))
        if fd:
            big = sorted(c["max_abs_diff"].items(), key=lambda kv: -kv[1])[:3]
            lines.append("                   largest per-column differences: %s" % ", ".join("%s %.4g" % kv for kv in big))
        lines.append("                   best.pt: selected epoch %s vs %s; %d of %d tensors bitwise identical (%s); max abs diff %.3g; train_metrics equal: %s" % (
            b["selected_epoch"][0], b["selected_epoch"][1], b["bitwise_identical"], b["tensors"], b["dtype"],
            b["max_abs_diff_float_tensors"], b["train_metrics_equal"]))
        if b["train_args_differences"]:
            lines.append("                   train_args differences: %s" % b["train_args_differences"])
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "g2_training.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "g2_training.json"))


if __name__ == "__main__":
    main()
