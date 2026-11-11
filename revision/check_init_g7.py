"""G7 pairing check (supplementary pre-registration S1, Section 4), CPU only.

For every seed of G7 (50-57), the baseline trainer is set up exactly as for training (YOLO
"yolov8s.pt" from the repository root, the LDPolypVideo data.yaml, 100 epochs, batch 16,
patience 30, deterministic=True, cache="disk", save_period=10, val=True) at imgsz 640 and at
imgsz 1280, and stopped at on_pretrain_routine_end, before the first training step. Only the
data-loader worker count differs (0 instead of 8, as in check13); it does not touch the model.
Every tensor of trainer.model.state_dict() is hashed (SHA-256 of its bytes, with dtype and
shape); only the hashes are kept, so memory stays low.

Result per seed: the number of tensors and the names of any tensor that differs between 640 and
1280. Control of the check: seed 50 against seed 51 at 640 (expected to differ only in the three
classification output convolutions model.22.cv3.{0,1,2}.2.weight, as in check13).
Rule (S1 Section 4): if all tensors are identical at both sizes for all eight seeds, the new
batch is analysed paired (Welch alongside); otherwise Welch is primary.
Writes only to --out (a new folder) and to the throw-away trainer folder --project.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone

EXPECT_CTRL = ["model.22.cv3.0.2.weight", "model.22.cv3.1.2.weight", "model.22.cv3.2.2.weight"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolov8s.pt", help="Relative to --repo, as in scripts/train.py")
    ap.add_argument("--expect-weights-sha256", required=True)
    ap.add_argument("--seeds", default="50,51,52,53,54,55,56,57")
    ap.add_argument("--sizes", default="640,1280")
    ap.add_argument("--project", default="/tmp/p3_initcheck_g7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    data = os.path.realpath(args.data)
    this_sha = sha256_file(os.path.abspath(__file__))
    os.chdir(args.repo)
    wsha = sha256_file(args.weights)
    if not wsha == args.expect_weights_sha256:
        sys.exit("ERROR: %s has SHA-256 %s, expected %s" % (args.weights, wsha, args.expect_weights_sha256))
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]

    import torch
    import ultralytics
    from ultralytics import YOLO

    hashes = {}
    t0 = time.time()
    for seed in seeds:
        for imgsz in sizes:
            model = YOLO(args.weights)
            box = {}

            def grab(trainer, box=box):
                box["h"] = {k: "%s|%s|%s" % (str(v.dtype), tuple(v.shape), hashlib.sha256(
                    v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k, v in trainer.model.state_dict().items()}
                raise Stop()

            model.add_callback("on_pretrain_routine_end", grab)
            try:
                model.train(data=data, epochs=100, imgsz=imgsz, batch=16, patience=30, workers=0, device="cpu",
                            seed=seed, deterministic=True, cache="disk", val=True, plots=False, save_period=10,
                            project=args.project, name="s%d_%d" % (seed, imgsz), exist_ok=True, verbose=False)
            except Stop:
                pass
            if "h" not in box:
                sys.exit("ERROR: seed %d imgsz %d: the trainer finished without reaching on_pretrain_routine_end" % (seed, imgsz))
            hashes[(seed, imgsz)] = box["h"]
            print("[%4.0f s] seed %d imgsz %d: %d tensors hashed; peak memory %.0f MB" % (
                time.time() - t0, seed, imgsz, len(box["h"]), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0), flush=True)
            del model, box
            gc.collect()

    def diff(a, b):
        A, B = hashes[a], hashes[b]
        if not sorted(A) == sorted(B):
            return None, sorted(set(A) ^ set(B))
        return len(A), [k for k in A if not A[k] == B[k]]

    per_seed = {}
    all_same = True
    for seed in seeds:
        n, d = diff((seed, sizes[0]), (seed, sizes[1]))
        same = n is not None and not d
        all_same = all_same and same
        per_seed[str(seed)] = {"tensors": n, "identical": same, "differing_tensors": d}
    ctrl = None
    if len(seeds) >= 2:
        n, d = diff((seeds[0], sizes[0]), (seeds[1], sizes[0]))
        ctrl = {"pair": "seed %d vs seed %d at imgsz %d" % (seeds[0], seeds[1], sizes[0]), "tensors": n,
                "differing_tensors": d, "as_expected": d == EXPECT_CTRL}
    if ctrl is None or not ctrl["differing_tensors"]:
        sys.exit("ERROR: the control pair shows no difference, so the check cannot tell seeds apart; nothing written")
    rule = "paired (Welch alongside)" if all_same else "Welch primary"
    rec = {"analysis": "G7 pairing check, supplementary pre-registration S1 Section 4",
           "run_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
           "settings": {"weights": os.path.abspath(args.weights), "weights_sha256": wsha, "data": data,
                        "seeds": seeds, "sizes": sizes, "device": "cpu", "workers": 0,
                        "stopped_at": "on_pretrain_routine_end", "compared": "SHA-256 of every tensor of trainer.model.state_dict()"},
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                        "platform": platform.platform()},
           "this_script_sha256": this_sha,
           "per_seed_640_vs_1280": per_seed, "control": ctrl, "all_seeds_identical": all_same,
           "rule_result_for_the_new_batch": rule}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "pairing_check_g7.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("")
    print("== G7 pairing check (S1 Section 4): imgsz %d vs %d, same seed" % (sizes[0], sizes[1]))
    for s, v in per_seed.items():
        print("  seed %s: %s tensors, %s" % (s, v["tensors"], "all identical" if v["identical"] else "DIFFERENT: %s" % v["differing_tensors"][:10]))
    if ctrl:
        print("== control of the check: %s: %d tensors differ %s; as expected (only the three cv3 output convolutions): %s" % (
            ctrl["pair"], len(ctrl["differing_tensors"]), ctrl["differing_tensors"][:10], ctrl["as_expected"]))
    print("== rule result for the new batch: %s" % rule)
    print("written:", os.path.join(args.out, "pairing_check_g7.json"))


if __name__ == "__main__":
    main()
