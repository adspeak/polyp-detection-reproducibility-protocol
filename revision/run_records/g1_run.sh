# Paper3 PeerJ revision, G1: one-time test/validation inference cache of the 17 replication
# checkpoints, the public evaluation scripts run once unchanged, and the exact-match check.
# Allowed only after the addendum v1.0 is deposited on Figshare (the DOI is required).
#   DOI=10.6084/m9.figshare.NNNNNNN bash g1_run.sh          # full run
#   DRY=1 DOI=test bash g1_run.sh                            # checks only, no inference
# The training queue is paused with $P3/QUEUE_PAUSE (created here if absent); G1 starts only
# after the queue reports that it is paused and no training process is left, and the pause
# file is removed again at the end (also on failure) if this script created it.
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
PY="${PY:-python}"
REPO="$P3/polyp-detection-reproducibility-protocol"
TOOLS="$P3/tools"
DS="$P3/data/ldpolyp"
THR="$P3/data/size_thresholds.npy"
EV="$P3/evaluation"
CACHE="$EV/cache/ldpolyp"
QLOG="${QLOG:-$P3/queue_20261105.log}"
PAUSE="$P3/QUEUE_PAUSE"
POLL="${POLL:-60}"
DRY="${DRY:-0}"
EXP_MS=3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559
EXP_CF=c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2
log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "失败：$*"; exit 1; }

[ -n "${DOI:-}" ] || fail "没有给 DOI（附录在 Figshare 发布后才能运行 G1）"
MS="$REPO/evaluation/multiseed_eval.py"; CF="$REPO/evaluation/crossover_fp_analysis.py"
[ "$(sha256sum "$MS" | cut -c1-64)" = "$EXP_MS" ] || fail "multiseed_eval.py 与公开版本不一致"
[ "$(sha256sum "$CF" | cut -c1-64)" = "$EXP_CF" ] || fail "crossover_fp_analysis.py 与公开版本不一致"
RUNS=""
for s in 42 43 44 45 46 47 48 49; do RUNS="$RUNS $P3/runs/baseline_seed$s"; done
for s in 42 43 44 45 46 47 48 49; do RUNS="$RUNS $P3/runs/p2_seed$s"; done
RUNS="$RUNS $P3/runs/control_seed42"
nbad=0
for r in $RUNS; do
  n=$(basename "$r")
  a=$(sha256sum < "$r/weights/best.pt" | cut -c1-64); b=$(sha256sum < "$P3/weights/$n.pt" | cut -c1-64)
  [ "$a" = "$b" ] || { log "weights/$n.pt 与 runs/$n/weights/best.pt 不一致"; nbad=$((nbad+1)); }
done
[ "$nbad" = 0 ] || fail "$nbad 个权重副本不一致"
log "检查通过：公开评估脚本哈希正确；17 个权重副本与 best.pt 一致；DOI=$DOI"

if [ "$DRY" = 1 ]; then
  "$PY" "$TOOLS/cache_predictions.py" --run-dirs $RUNS --dataset "$DS" --splits test val --out "$CACHE" --doi "$DOI" --dry-run
  log "试运行结束（退出码 $?），没有推理"
  exit 0
fi

nvidia-smi -L 2>/dev/null | grep -q "^GPU " || fail "没有 GPU"
CREATED=0
if [ -f "$PAUSE" ]; then log "$PAUSE 已存在（不是本脚本创建的）"; else touch "$PAUSE"; CREATED=1; log "已创建 $PAUSE，等当前训练结束后队列暂停"; fi
cleanup() { if [ "$CREATED" = 1 ] && [ -f "$PAUSE" ]; then rm -f "$PAUSE"; log "已删除 $PAUSE，训练队列继续"; fi; }
trap cleanup EXIT
trap 'log "收到中断信号"; exit 130' INT TERM

# wait: the queue has logged that it is paused (or has finished, or is not running), and no training process is left
t0=$(date +%s); last=0
while :; do
  qline=$(grep -a -E "^\[20" "$QLOG" 2>/dev/null | tail -1)
  qrun=$(pgrep -f "^[^ ]*bash .*queue_g2g3g5[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
  trun=$(pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null && echo 1 || echo 0)
  if [ "$trun" = 0 ] && { [ "$qrun" = 0 ] || echo "$qline" | grep -q -E "QUEUE_PAUSE，暂停|队列全部结束"; }; then break; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 600 ]; then log "等待训练队列暂停（已等 $(( (now - t0) / 60 )) 分钟）；队列最后一行：${qline:-无}"; last=$now; fi
  sleep "$POLL"
done
log "训练队列已暂停（或未运行），没有训练进程；开始 G1"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU 状态: /'

STAMP=$(date +%Y%m%d_%H%M%S)
PUB="$EV/public_check/$STAMP"
mkdir -p "$PUB"
log "1/3 缓存推理：17 个模型 × test、val（conf 0.001）"
"$PY" "$TOOLS/cache_predictions.py" --run-dirs $RUNS --dataset "$DS" --splits test val --out "$CACHE" --doi "$DOI" \
  || fail "缓存推理出错"

log "2/3 原样运行两个公开评估脚本（conf 0.25）"
B=""; C=""; for s in 42 43 44 45 46 47 48 49; do B="$B baseline_seed$s.pt"; C="$C p2_seed$s.pt"; done
"$PY" - "$PUB/jobs.json" "$P3" "$DS" "$THR" "$MS" "$CF" "$B" "$C" <<'PYEOF'
import hashlib, json, sys
out, p3, ds, thr, ms, cf, b, c = sys.argv[1:9]
b, c = b.split(), c.split()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
seeds = ["42", "43", "44", "45", "46", "47", "48", "49"]
common = ["--dataset", ds, "--weights-dir", p3 + "/weights", "--imgsz", "1280", "--conf", "0.25", "--iou-match", "0.5"]
pub = out.rsplit("/", 1)[0]
jobs = [
  {"name": "multiseed_p2", "script": "multiseed", "path": ms, "argv": common + ["--thresholds", thr, "--baseline"] + b + ["--candidate"] + c + ["--seeds"] + seeds},
  {"name": "crossover_p2", "script": "crossover", "path": cf, "argv": common + ["--baseline"] + b + ["--candidate"] + c + ["--seeds"] + seeds},
  {"name": "multiseed_control", "script": "multiseed", "path": ms, "argv": common + ["--thresholds", thr, "--baseline", "baseline_seed42.pt", "--candidate", "control_seed42.pt", "--seeds", "42"]},
  {"name": "crossover_control", "script": "crossover", "path": cf, "argv": common + ["--baseline", "baseline_seed42.pt", "--candidate", "control_seed42.pt", "--seeds", "42"]},
]
for j in jobs:
    j["out"] = "%s/%s.json" % (pub, j["name"])
    j["argv"] += ["--out", j["out"]]
    j["script_sha256"] = sha(j["path"])
json.dump(jobs, open(out, "w"), indent=2)
print("\n".join(j["name"] for j in jobs))
PYEOF
[ -f "$PUB/jobs.json" ] || fail "没写出 jobs.json"
for j in multiseed_p2 crossover_p2 multiseed_control crossover_control; do
  args=$("$PY" -c "import json,sys,shlex; j=[x for x in json.load(open(sys.argv[1])) if x['name']==sys.argv[2]][0]; print(shlex.quote(j['path']), ' '.join(shlex.quote(a) for a in j['argv']))" "$PUB/jobs.json" "$j")
  log "  运行 $j"
  ( cd "$PUB" && eval "\"$PY\" $args" ) > "$PUB/$j.log" 2>&1 || fail "$j 出错，见 $PUB/$j.log"
  tail -3 "$PUB/$j.log" | sed 's/^/    /'
done

log "3/3 用缓存重放公开脚本，逐个数值对比"
"$PY" "$TOOLS/check_public_vs_cache.py" --repo "$REPO" --cache "$CACHE" --split test --jobs "$PUB/jobs.json" --out "$PUB/replay"
rc=$?
if [ "$rc" = 0 ]; then log "G1 完成：缓存与公开脚本的结果完全一致"; else log "G1 对比不一致（退出码 $rc），先排查，暂停后续分析"; fi
log "结果目录：$PUB；缓存目录：$CACHE"
[ "$CREATED" = 1 ] || log "注意：$PAUSE 不是本脚本创建的，没有删除；确认后手动 rm 让队列继续"
exit "$rc"
