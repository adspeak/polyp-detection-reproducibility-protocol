# Paper3 PeerJ revision: one-time test/validation cache of the G3 (640) and G2 checkpoints,
# inserted into the training queue between the last G3 run and the first G5 run.
#   DOI=10.6084/m9.figshare.34919340 bash g3_cache.sh
# Waits until the queue has started baseline640_seed49, creates $P3/QUEUE_PAUSE, waits until the
# queue reports that it is paused and no training process is left, caches the 12 checkpoints,
# and removes the pause file again (also on failure or interruption) if this script created it.
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
PY="${PY:-python}"
QLOG="${QLOG:-$P3/queue_20261105.log}"
PAUSE="$P3/QUEUE_PAUSE"
POLL="${POLL:-30}"
log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "失败：$*"; exit 1; }
[ -n "${DOI:-}" ] || fail "没有给 DOI"
RUNS=""
for s in 42 43 44 45 46 47 48 49; do RUNS="$RUNS $P3/runs_g3/baseline640_seed$s"; done
for n in baseline_seed42 p2_seed42 baseline_seed43 p2_seed43; do RUNS="$RUNS $P3/runs_g2/$n"; done
log "等待队列开始 baseline640_seed49（G3 的最后一个 run）"
t0=$(date +%s); last=0
until grep -a -q -E "开始 (baseline640_seed49|control_seed4[3-9])|队列全部结束" "$QLOG"; do
  now=$(date +%s)
  if [ $((now - last)) -ge 600 ]; then log "仍在等待；队列最后一行：$(grep -a -E '^\[20' "$QLOG" | tail -1)"; last=$now; fi
  sleep "$POLL"
done
CREATED=0
if [ -f "$PAUSE" ]; then log "$PAUSE 已存在（不是本脚本创建的）"; else touch "$PAUSE"; CREATED=1; log "已创建 $PAUSE"; fi
cleanup() { if [ "$CREATED" = 1 ] && [ -f "$PAUSE" ]; then rm -f "$PAUSE"; log "已删除 $PAUSE，训练队列继续"; fi; }
trap cleanup EXIT
trap 'log "收到中断信号"; exit 130' INT TERM
last=0
while :; do
  qline=$(grep -a -E "^\[20" "$QLOG" 2>/dev/null | tail -1)
  qrun=$(pgrep -f "^[^ ]*bash .*queue_g2g3g5[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
  trun=$(pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null && echo 1 || echo 0)
  if [ "$trun" = 0 ] && { [ "$qrun" = 0 ] || echo "$qline" | grep -q -E "QUEUE_PAUSE，暂停|队列全部结束"; }; then break; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 600 ]; then log "等待当前训练结束、队列暂停；队列最后一行：${qline:-无}"; last=$now; fi
  sleep "$POLL"
done
log "队列已暂停，没有训练进程；开始缓存推理（G3 8 个 + G2 4 个模型 × test、val）"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU 状态: /'
"$PY" "$P3/tools/cache_predictions.py" --run-dirs $RUNS --dataset "$P3/data/ldpolyp" --splits test val \
  --out "$P3/evaluation/cache/ldpolyp" --doi "$DOI" || fail "缓存推理出错"
log "缓存完成：test $(ls "$P3"/evaluation/cache/ldpolyp/test/*.npz | wc -l) 个 npz，val $(ls "$P3"/evaluation/cache/ldpolyp/val/*.npz | wc -l) 个 npz"
