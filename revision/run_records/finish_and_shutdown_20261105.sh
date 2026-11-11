# Paper3 PeerJ revision: after the last training run of tonight, cache the new checkpoints once,
# check the 640 caches end to end against the public scripts, then shut the AutoDL instance down
# (/usr/bin/shutdown, as in the AutoDL docs).
#   nohup bash finish_and_shutdown_20261105.sh > $P3/finish_20261105.log 2>&1 &
# 1. waits until neither queue script (G2/G3/G5, G6) nor any train_paper3.py process is running;
# 2. one-time inference cache (test and val, conf 0.001, as in G1) of the finished G5 runs
#    (runs/control_seed43-49, LDPolypVideo) and G6 runs (runs_g6, Kvasir-SEG data_v3);
#    existing caches are never regenerated;
# 3. 640 exact-match check (added after the addendum; no analysis uses its output): the public
#    multiseed_eval.py and crossover_fp_analysis.py are run unchanged at imgsz 640, conf 0.25, on
#    the G3 checkpoints (LDPolypVideo, 42-45 against 46-49) and on the G6 checkpoints (Kvasir-SEG,
#    baseline against +P2), and replayed from the cache with check_public_vs_cache.py as in G1;
#    only "identical / different" is printed here, the numbers stay in the output folders;
# 4. shuts down, unless $P3/NO_SHUTDOWN exists (touch it to cancel; or kill this script).
# A failure in steps 2-3 is logged and does not stop step 4.
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
PY="${PY:-python}"
POLL="${POLL:-60}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/usr/bin/shutdown}"
REPO="$P3/polyp-detection-reproducibility-protocol"
EV="$P3/evaluation/cache"
MS="$REPO/evaluation/multiseed_eval.py"; CF="$REPO/evaluation/crossover_fp_analysis.py"
EXP_MS=3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559
EXP_CF=c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2
log() { echo "[$(date '+%F %T')] $*"; }

log "等待训练全部结束（G2/G3/G5 队列、G6 队列、train_paper3.py 都不在）"
t0=$(date +%s); last=0
while :; do
  q5=$(pgrep -f "^[^ ]*bash .*queue_g2g3g5[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
  q6=$(pgrep -f "^[^ ]*bash .*queue_g6[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
  tr=$(pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null && echo 1 || echo 0)
  if [ "$q5$q6$tr" = 000 ]; then break; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 1800 ]; then log "仍在训练（已等 $(( (now - t0) / 60 )) 分钟）；G6 最后一行：$(grep -a -E '^\[20' "$P3/queue_g6_20261105.log" 2>/dev/null | tail -1)"; last=$now; fi
  sleep "$POLL"
done
log "训练全部结束。G5 日志最后一行：$(grep -a -E '^\[20' "$P3/queue_20261105.log" 2>/dev/null | tail -1)"
log "G6 日志最后一行：$(grep -a -E '^\[20' "$P3/queue_g6_20261105.log" 2>/dev/null | tail -1)"
sleep "${SETTLE:-30}"

# ---------- 2. one-time caches ----------
C5=0; C6=0
DOI=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['addendum_doi'])" "$EV/ldpolyp/test/runs_g3__baseline640_seed42.json" 2>/dev/null)
if [ -z "$DOI" ]; then log "读不到附录 DOI（G3 缓存记录），跳过缓存和核对"; else
  R5=""; n5=0; for s in 43 44 45 46 47 48 49; do d="$P3/runs/control_seed$s"
    if [ -f "$d/recovery_metadata.json" ]; then R5="$R5 $d"; n5=$((n5+1)); else log "跳过未完成的 $d"; fi; done
  R6=""; n6=0; for s in 42 43 44 45 46 47 48 49; do for c in baseline p2; do d="$P3/runs_g6/${c}_seed$s"
    if [ -f "$d/recovery_metadata.json" ]; then R6="$R6 $d"; n6=$((n6+1)); else log "跳过未完成的 $d"; fi; done; done
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU 状态: /'
  if [ "$n5" -gt 0 ]; then
    log "缓存 G5：$n5 个模型 × test、val（LDPolypVideo）"
    if "$PY" "$P3/tools/cache_predictions.py" --run-dirs $R5 --dataset "$P3/data/ldpolyp" --splits test val --out "$EV/ldpolyp" --doi "$DOI"; then
      C5=1; log "G5 缓存完成"; else log "G5 缓存出错（见上面），继续"; fi
  fi
  if [ "$n6" -gt 0 ]; then
    log "缓存 G6：$n6 个模型 × test、val（Kvasir-SEG data_v3）"
    if "$PY" "$P3/tools/cache_predictions.py" --run-dirs $R6 --dataset "$P3/data/kvasir_v3_yolo" --splits test val --out "$EV/kvasir_v3" --doi "$DOI"; then
      C6=1; log "G6 缓存完成"; else log "G6 缓存出错（见上面），继续"; fi
  fi
  log "缓存文件数：ldpolyp test $(ls "$EV"/ldpolyp/test/*.npz 2>/dev/null | wc -l)，val $(ls "$EV"/ldpolyp/val/*.npz 2>/dev/null | wc -l)；kvasir_v3 test $(ls "$EV"/kvasir_v3/test/*.npz 2>/dev/null | wc -l)，val $(ls "$EV"/kvasir_v3/val/*.npz 2>/dev/null | wc -l)"
fi

# ---------- 3. 640 exact-match check against the public scripts ----------
STAMP=$(date +%Y%m%d_%H%M%S)
check640() {  # label dataset cache_root thresholds seeds_csv baseline_csv candidate_csv
  local label=$1 ds=$2 cache=$3 thr=$4 seeds=$5 B=$6 C=$7
  local pub="$P3/evaluation/public_check_640/${STAMP}_$label"
  mkdir -p "$pub"
  "$PY" - "$pub/jobs.json" "$P3" "$ds" "$thr" "$MS" "$CF" "$seeds" "$B" "$C" <<'PYEOF' || { log "  $label：写 jobs.json 出错"; return 1; }
import hashlib, json, sys
import numpy as np
out, p3, ds, thr, ms, cf, seeds, b, c = sys.argv[1:10]
seeds, b, c = seeds.split(","), b.split(","), c.split(",")
t = np.load(thr, allow_pickle=True).item()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
common = ["--dataset", ds, "--weights-dir", p3, "--imgsz", "640", "--conf", "0.25", "--iou-match", "0.5"]
pub = out.rsplit("/", 1)[0]
jobs = [{"name": "multiseed", "script": "multiseed", "path": ms,
         "argv": common + ["--thresholds", thr, "--baseline"] + b + ["--candidate"] + c + ["--seeds"] + seeds},
        {"name": "crossover", "script": "crossover", "path": cf,
         "argv": common + ["--p33", repr(float(t["p33"])), "--p67", repr(float(t["p67"])), "--baseline"] + b + ["--candidate"] + c + ["--seeds"] + seeds}]
for j in jobs:
    j["out"] = "%s/%s.json" % (pub, j["name"])
    j["argv"] += ["--out", j["out"]]
    j["script_sha256"] = sha(j["path"])
json.dump(jobs, open(out, "w"), indent=2)
PYEOF
  local j args
  for j in multiseed crossover; do
    args=$("$PY" -c "import json,sys,shlex; j=[x for x in json.load(open(sys.argv[1])) if x['name']==sys.argv[2]][0]; print(shlex.quote(j['path']), ' '.join(shlex.quote(a) for a in j['argv']))" "$pub/jobs.json" "$j")
    log "  $label：原样运行公开脚本 $j（真实推理，640）"
    ( cd "$pub" && eval "\"$PY\" $args" ) > "$pub/$j.log" 2>&1 || { log "  $label：$j 出错，见 $pub/$j.log"; return 1; }
  done
  "$PY" "$P3/tools/check_public_vs_cache.py" --repo "$REPO" --cache "$cache" --split test --jobs "$pub/jobs.json" --out "$pub/replay" > "$pub/check.log" 2>&1
  local rc=$?
  grep -E "identical|DIFFERENT|RESULT|exactly 0.25|images requested" "$pub/check.log" | sed "s/^/    /"
  log "  $label：核对退出码 $rc，结果目录 $pub"
  return $rc
}
if [ "$(sha256sum "$MS" | cut -c1-64)" = "$EXP_MS" ] && [ "$(sha256sum "$CF" | cut -c1-64)" = "$EXP_CF" ]; then
  log "640 核对（附录之外新增的端到端检查）：公开脚本哈希正确"
  B3=""; C3=""; for s in 42 43 44 45; do B3="$B3,runs_g3/baseline640_seed$s/weights/best.pt"; C3="$C3,runs_g3/baseline640_seed$((s+4))/weights/best.pt"; done
  check640 g3_ldpolyp "$P3/data/ldpolyp" "$EV/ldpolyp" "$P3/data/size_thresholds.npy" "42v46,43v47,44v48,45v49" "${B3#,}" "${C3#,}" \
    && log "G3（LDPolypVideo，640）：缓存与公开脚本完全一致" || log "G3（LDPolypVideo，640）：核对没通过，先排查，暂停相关分析"
  if [ "$C6" = 1 ]; then
    B6=""; C6L=""; S6=""; for s in 42 43 44 45 46 47 48 49; do
      if [ -f "$P3/runs_g6/baseline_seed$s/recovery_metadata.json" ] && [ -f "$P3/runs_g6/p2_seed$s/recovery_metadata.json" ]; then
        B6="$B6,runs_g6/baseline_seed$s/weights/best.pt"; C6L="$C6L,runs_g6/p2_seed$s/weights/best.pt"; S6="$S6,$s"; fi; done
    if [ -n "$S6" ]; then
      check640 g6_kvasir "$P3/data/kvasir_v3_yolo" "$EV/kvasir_v3" "$P3/data/kvasir_v3_yolo/size_thresholds.npy" "${S6#,}" "${B6#,}" "${C6L#,}" \
        && log "G6（Kvasir-SEG，640）：缓存与公开脚本完全一致" || log "G6（Kvasir-SEG，640）：核对没通过，先排查，暂停相关分析"
    else log "G6 没有成对完成的种子，跳过 Kvasir 核对"; fi
  else log "G6 缓存没有完成，跳过 Kvasir 核对"; fi
else log "公开脚本哈希不对，跳过 640 核对"; fi

# ---------- 4. shut down ----------
df -h "$P3" | tail -1 | sed 's/^/  磁盘: /'
sync
if [ -f "$P3/NO_SHUTDOWN" ]; then log "检测到 $P3/NO_SHUTDOWN，不关机"; exit 0; fi
log "关机：$SHUTDOWN_CMD"
sync
$SHUTDOWN_CMD
