# Paper3 PeerJ revision: training queue G7, the positive-control extension registered in the
# supplementary pre-registration S1 v1.0 (DOI 10.6084/m9.figshare.34933821). Trains only; never
# evaluates the test split. Baseline YOLOv8s on LDPolypVideo, seeds 50-57, each at imgsz 640 and
# then at 1280, into $P3/runs_g7, in the order of S1 Section 2. The environment must match the G3
# runs (S1 Section 3): code commit, train_paper3.py, yolov8s.pt and tracked-file state are compared
# with the launch record of runs_g3/baseline640_seed42, and the software versions with S1.
# A failed run is restarted once with the same seed and settings; if it fails again its pair is
# reported as missing (S1 Section 2). No seed is added or replaced.
#   DRY=1 bash queue_g7_20261106.sh   -> preflight + data checks + planned runs, no training
#   touch $P3/QUEUE_PAUSE              -> the queue waits before starting the next run; rm it to resume
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
REPO="$P3/polyp-detection-reproducibility-protocol"
TRAIN="$P3/tools/train_paper3.py"
LD="$P3/data/ldpolyp"
DATA="$LD/data.yaml"
REC="$P3/launch_records"
G3REC="$REC/runs_g3__baseline640_seed42.json"
EXPECT_N="${EXPECT_N:-train:2118 val:402 test:1570}"
EXPECT_SW="${EXPECT_SW:-3.8.10 2.0.0+cu118 8.4.104 8700}"
SEEDS="${SEEDS:-50 51 52 53 54 55 56 57}"
DRY="${DRY:-0}"
log() { echo "[$(date '+%F %T')] $*"; }
free_gb() { df -BG --output=avail "$P3" | tail -1 | tr -dc 0-9; }
die() { log "$*，退出"; exit 1; }

# ---------- preflight: GPU, disk, no other training ----------
if nvidia-smi -L 2>/dev/null | grep -q "^GPU "; then log "GPU: $(nvidia-smi -L | head -1)"; else
  if [ "$DRY" = 1 ]; then log "GPU: 没有（试运行，不影响）"; else die "没有 GPU"; fi
fi
[ "$(free_gb)" -ge 5 ] || die "剩余空间 $(free_gb)G，不足 5G"
if pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null; then
  if [ "$DRY" = 1 ]; then log "注意：现在有 train_paper3.py 在运行（试运行，不影响）"; else die "已有 train_paper3.py 在运行"; fi
fi

# ---------- environment = G3 (S1 Section 3) ----------
[ -f "$G3REC" ] || die "没有 G3 的启动记录 $G3REC"
W_SHA=$(sha256sum "$REPO/yolov8s.pt" | cut -c1-64)
T_SHA=$(sha256sum "$TRAIN" | cut -c1-64)
COMMIT=$(git -C "$REPO" rev-parse HEAD)
TAGS=$(git -C "$REPO" tag --points-at HEAD | tr '\n' ' ')
TRACKED_DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)
G3=$(python - "$G3REC" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(r["yolov8s_sha256"], r["train_paper3_sha256"], r["code_commit"], r["tracked_dirty_lines"])
PYEOF
)
set -- $G3
[ "$W_SHA" = "$1" ] || die "yolov8s.pt 的哈希 ${W_SHA:0:16} 与 G3 的 ${1:0:16} 不同"
[ "$T_SHA" = "$2" ] || die "train_paper3.py 的哈希 ${T_SHA:0:16} 与 G3 的 ${2:0:16} 不同"
[ "$COMMIT" = "$3" ] || die "代码 commit ${COMMIT:0:7} 与 G3 的 ${3:0:7} 不同"
[ "$TRACKED_DIRTY" = "$4" ] || die "已跟踪文件改动 $TRACKED_DIRTY 行，与 G3 的 $4 行不同"
SW=$(python -c "import sys, torch, ultralytics; print(sys.version.split()[0], torch.__version__, ultralytics.__version__, torch.backends.cudnn.version())" 2>/dev/null)
if [ "$SW" = "$EXPECT_SW" ]; then :; else
  if [ "$DRY" = 1 ] && [ "${SW% *}" = "${EXPECT_SW% *}" ]; then log "注意：cuDNN 报告为 ${SW##* }（没有 GPU 时可能如此；试运行，不影响）"
  else die "软件版本是 [$SW]，S1 规定 [$EXPECT_SW]"; fi
fi
HOST=$(hostname); KERN=$(uname -r)
DRV=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1)
log "preflight 通过：与 G3 相同（commit=${COMMIT:0:7} tags=[$TAGS] 已跟踪文件改动=$TRACKED_DIRTY 行 train_paper3.py=${T_SHA:0:16} yolov8s.pt=${W_SHA:0:16}）；软件 [$SW]；主机 $HOST，驱动 [${DRV:-无}]，内核 $KERN；剩余 $(free_gb)G"

# ---------- data: LDPolypVideo as used by G3 and the replication ----------
grep -q -x "path: $LD" "$DATA" 2>/dev/null || die "$DATA 不存在或 path 不是 $LD"
nbad=0
for sp in $EXPECT_N; do
  s=${sp%%:*}; n=${sp##*:}
  ni=$(ls "$LD/images/$s" 2>/dev/null | grep -c '\.jpg$'); nl=$(ls "$LD/labels/$s" 2>/dev/null | grep -c '\.txt$')
  if [ "$ni" = "$n" ] && [ "$nl" = "$n" ]; then :; else log "$s: jpg $ni、标签 $nl，应为 $n"; nbad=$((nbad+1)); fi
done
[ -f "$P3/data/size_thresholds.npy" ] || { log "没有 $P3/data/size_thresholds.npy"; nbad=$((nbad+1)); }
[ "$nbad" = 0 ] || die "数据检查没通过"
log "数据检查通过：$LD（$EXPECT_N），阈值文件存在"

mkdir -p "$REC" "$P3/env"
if [ "$DRY" = 1 ]; then log "试运行：不写环境存档"; else
  STAMP=$(date +%Y%m%d_%H%M)
  pip freeze > "$P3/env/pip_freeze_g7_$STAMP.txt"
  { echo "host $HOST"; echo "kernel $KERN"; echo "driver_gpu $DRV"; echo "software $SW"; echo "commit $COMMIT tags $TAGS"
    echo "train_paper3_sha256 $T_SHA"; echo "yolov8s_sha256 $W_SHA"; echo "s1_doi 10.6084/m9.figshare.34933821"; } > "$P3/env/host_g7_$STAMP.txt"
  log "环境存档：env/pip_freeze_g7_$STAMP.txt、env/host_g7_$STAMP.txt"
fi

run_one() {  # project name config seed imgsz; returns 0 if the run is complete
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
  local rj="$REC/$(basename "$project")__$name.json"
  [ -f "$rj" ] && mv "$rj" "$rj.failed_$(date +%Y%m%d_%H%M%S)"
  printf '{"name": "%s", "project": "%s", "config": "%s", "seed": %s, "imgsz": %s, "exit_code": %s, "status": "%s", "seconds": %s, "epoch_files_deleted": %s, "epoch_bytes_deleted": %s, "best_sha256": "%s", "train_paper3_sha256": "%s", "yolov8s_sha256": "%s", "code_commit": "%s", "git_tags": "%s", "tracked_dirty_lines": %s, "host": "%s", "driver_gpu": "%s"}\n' \
    "$name" "$project" "$cfg" "$seed" "$imgsz" "$rc" "$status" "$(( $(date +%s) - t0 ))" "$n_ep" "$b_ep" \
    "$best_sha" "$T_SHA" "$W_SHA" "$COMMIT" "$TAGS" "$TRACKED_DIRTY" "$HOST" "$DRV" > "$rj"
  log "结束 $name：$status，退出码 $rc，用时 $(( ($(date +%s) - t0) / 60 )) 分钟，删除中间权重 $n_ep 个"
  [ "$status" = done ]
}

FAILED=""
run_s1() {  # S1 Section 2: one restart with the same seed and settings
  run_one "$@" && return 0
  log "$2 失败：按 S1 第 2 节用同一种子和设置重跑一次"
  run_one "$@" && return 0
  log "$2 两次都失败：这一对按缺失处理，不换种子"
  FAILED="$FAILED $2"
}

# ---------- G7: seeds 50-57, 640 then 1280 for each seed (S1 Section 2) ----------
for s in $SEEDS; do
  run_s1 "$P3/runs_g7" "baseline640_seed$s" baseline "$s" 640
  run_s1 "$P3/runs_g7" "baseline1280_seed$s" baseline "$s" 1280
done
if [ "$DRY" = 1 ]; then log "试运行结束：以上是正式运行时的顺序"; else log "G7 队列全部结束（两次都失败的 run：${FAILED:- 无}）"; fi
