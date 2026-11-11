# Paper3 PeerJ revision: training queue G6 (Kvasir-SEG, data_v3 roles, addendum v1.0 Section 11).
# Trains only; never evaluates the test split. Starts only after the G2/G3/G5 queue has logged
# 「队列全部结束」 and no training process is left; exits without training if that queue stopped
# any other way. Baseline and +P2, seeds 42-49, imgsz 640, interleaved by seed, into $P3/runs_g6.
# DRY=1 bash queue_g6_20261105.sh   -> preflight + data checks + planned runs, no waiting, no training
# touch $P3/QUEUE_PAUSE -> the queue waits before starting the next run; rm it to resume
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
REPO="$P3/polyp-detection-reproducibility-protocol"
TRAIN="$P3/tools/train_paper3.py"
KV="$P3/data/kvasir_v3_yolo"
DATA="$KV/data.yaml"
REC="$P3/launch_records"
G5LOG="${G5LOG:-$P3/queue_20261105.log}"
POLL="${POLL:-60}"
EXPECT_W="${EXPECT_W:-1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a}"
DRY="${DRY:-0}"
log() { echo "[$(date '+%F %T')] $*"; }
free_gb() { df -BG --output=avail "$P3" | tail -1 | tr -dc 0-9; }

# ---------- preflight ----------
if nvidia-smi -L 2>/dev/null | grep -q "^GPU "; then log "GPU: $(nvidia-smi -L | head -1)"; else
  if [ "$DRY" = 1 ]; then log "GPU: 没有（试运行，不影响）"; else log "没有 GPU，退出"; exit 1; fi
fi
[ "$(free_gb)" -ge 5 ] || { log "剩余空间 $(free_gb)G，不足 5G，退出"; exit 1; }
W_SHA=$(sha256sum "$REPO/yolov8s.pt" | cut -c1-64)
[ "$W_SHA" = "$EXPECT_W" ] || { log "repo 里 yolov8s.pt 的哈希不对: $W_SHA，退出"; exit 1; }
T_SHA=$(sha256sum "$TRAIN" | cut -c1-64)
TAGS=$(git -C "$REPO" tag --points-at HEAD | tr '\n' ' ')
COMMIT=$(git -C "$REPO" rev-parse HEAD)
TRACKED_DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)
log "preflight 通过：commit=${COMMIT:0:7} tags=[$TAGS] 已跟踪文件改动=$TRACKED_DIRTY 行 train_paper3.py=${T_SHA:0:16} yolov8s.pt=${W_SHA:0:16} 剩余 $(free_gb)G"

# ---------- data: the Kvasir-SEG dataset built from the data_v3 roles (check14) ----------
grep -q -x "path: $KV" "$DATA" 2>/dev/null || { log "$DATA 不存在或 path 不是 $KV，退出"; exit 1; }
nbad=0
for sp in train:700 val:150 test:150; do
  s=${sp%%:*}; n=${sp##*:}
  ni=$(ls "$KV/images/$s" 2>/dev/null | wc -l); nl=$(ls "$KV/labels/$s" 2>/dev/null | wc -l)
  if [ "$ni" = "$n" ] && [ "$nl" = "$n" ]; then :; else log "$s: 图像 $ni、标签 $nl，应为 $n"; nbad=$((nbad+1)); fi
done
[ -f "$KV/size_thresholds.npy" ] || { log "没有 $KV/size_thresholds.npy"; nbad=$((nbad+1)); }
[ "$nbad" = 0 ] || { log "数据检查没通过，退出"; exit 1; }
log "数据检查通过：$KV，train 700 / val 150 / test 150，阈值文件存在"

# ---------- wait for the G2/G3/G5 queue to finish ----------
if [ "$DRY" = 1 ]; then
  log "试运行：正式运行时先等 G2/G3/G5 队列结束（$G5LOG 出现「队列全部结束」且没有训练进程）"
else
  t0=$(date +%s); last=0
  while :; do
    qrun=$(pgrep -f "^[^ ]*bash .*queue_g2g3g5[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
    trun=$(pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null && echo 1 || echo 0)
    if [ "$qrun" = 0 ] && [ "$trun" = 0 ]; then break; fi
    now=$(date +%s)
    if [ $((now - last)) -ge 600 ]; then log "等待 G2/G3/G5 队列结束（已等 $(( (now - t0) / 60 )) 分钟）；它的最后一行：$(grep -a -E '^\[20' "$G5LOG" | tail -1)"; last=$now; fi
    sleep "$POLL"
  done
  if grep -a -q "队列全部结束" "$G5LOG"; then log "G2/G3/G5 队列已正常结束，没有训练进程；开始 G6"
  else log "G2/G3/G5 队列进程已不在，但日志里没有「队列全部结束」，G6 不启动，退出"; exit 1; fi
fi

mkdir -p "$REC" "$P3/env"
if [ "$DRY" = 1 ]; then log "试运行：不写环境存档"; else pip freeze > "$P3/env/pip_freeze_$(date +%Y%m%d_%H%M).txt"; fi

run_one() {  # project name config seed imgsz
  local project=$1 name=$2 cfg=$3 seed=$4 imgsz=$5
  local dir="$project/$name"
  if [ -f "$dir/recovery_metadata.json" ]; then log "已完成，跳过: $dir"; return 0; fi
  if [ "$DRY" = 1 ]; then
    if [ -d "$dir" ]; then log "计划: $name（$cfg seed=$seed imgsz=$imgsz）→ $project，目录已存在但未完成，正式运行时会改名保留后重训"
    else log "计划: $name（$cfg seed=$seed imgsz=$imgsz）→ $project"; fi
    return 0
  fi
  if [ -f "$P3/QUEUE_PAUSE" ]; then log "检测到 $P3/QUEUE_PAUSE，暂停；删除该文件后继续"; while [ -f "$P3/QUEUE_PAUSE" ]; do sleep 60; done; log "继续"; fi
  if [ "$(free_gb)" -lt 3 ]; then log "剩余空间不足 3G，停止队列"; exit 1; fi
  if [ -d "$dir" ]; then local keep="$dir.interrupted_$(date +%Y%m%d_%H%M%S)"; mv "$dir" "$keep"; log "未完成的旧目录改名保留: $keep"; fi
  local t0; t0=$(date +%s)
  log "开始 $name（$cfg seed=$seed imgsz=$imgsz → $project）"
  python "$TRAIN" --repo "$REPO" --config "$cfg" --seed "$seed" --data "$DATA" \
    --epochs 100 --imgsz "$imgsz" --batch 16 --project "$project" --name "$name"
  local rc=$?
  local n_ep b_ep status best_sha=""
  n_ep=$(ls "$dir"/weights/epoch*.pt 2>/dev/null | wc -l)
  b_ep=$(du -cb "$dir"/weights/epoch*.pt 2>/dev/null | tail -1 | cut -f1); b_ep=${b_ep:-0}
  if [ "$rc" = 0 ] && [ -f "$dir/recovery_metadata.json" ]; then
    rm -f "$dir"/weights/epoch*.pt; status=done
  else
    status=failed; n_ep=0; b_ep=0
  fi
  [ -f "$dir/weights/best.pt" ] && best_sha=$(sha256sum "$dir/weights/best.pt" | cut -c1-64)
  printf '{"name": "%s", "project": "%s", "config": "%s", "seed": %s, "imgsz": %s, "exit_code": %s, "status": "%s", "seconds": %s, "epoch_files_deleted": %s, "epoch_bytes_deleted": %s, "best_sha256": "%s", "train_paper3_sha256": "%s", "yolov8s_sha256": "%s", "code_commit": "%s", "git_tags": "%s", "tracked_dirty_lines": %s}\n' \
    "$name" "$project" "$cfg" "$seed" "$imgsz" "$rc" "$status" "$(( $(date +%s) - t0 ))" "$n_ep" "$b_ep" \
    "$best_sha" "$T_SHA" "$W_SHA" "$COMMIT" "$TAGS" "$TRACKED_DIRTY" > "$REC/$(basename "$project")__$name.json"
  log "结束 $name：$status，退出码 $rc，用时 $(( ($(date +%s) - t0) / 60 )) 分钟，删除中间权重 $n_ep 个"
}

# ---------- G6: Kvasir-SEG, baseline and +P2, seeds 42-49, imgsz 640 ----------
for s in 42 43 44 45 46 47 48 49; do
  run_one "$P3/runs_g6" "baseline_seed$s" baseline "$s" 640
  run_one "$P3/runs_g6" "p2_seed$s" p2 "$s" 640
done
if [ "$DRY" = 1 ]; then log "试运行结束：以上 16 个 run 是正式运行时的顺序"; else log "G6 队列全部结束"; fi
