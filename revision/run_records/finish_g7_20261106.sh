# Paper3 PeerJ revision: after the G7 queue (S1 v1.0, DOI 10.6084/m9.figshare.34933821), cache the
# G7 checkpoints once, check the caches end to end against the public scripts, then shut the AutoDL
# instance down (/usr/bin/shutdown).
#   nohup bash finish_g7_20261106.sh > $P3/finish_g7_20261106.log 2>&1 &
# 1. waits until neither queue_g7 nor any train_paper3.py process is running;
#    if the queue log has no 「G7 队列全部结束」 line, nothing is cached or checked (step 4 still runs);
# 2. one-time inference cache (S1 Section 5; test and val, conf 0.001, each model at its training
#    input size, the tool of G1) of every finished G7 run; existing caches are never regenerated;
# 3. end-to-end check (S1 Section 5, verification only): the public multiseed_eval.py and
#    crossover_fp_analysis.py run unchanged at imgsz 640 on the eight 640 models (seeds 50-53 against
#    54-57) and at 1280 on the eight 1280 models (same split), conf 0.25, then replayed from the cache
#    with check_public_vs_cache.py; only "identical / different" is printed here;
# 4. shuts down, unless $P3/NO_SHUTDOWN exists (touch it to cancel; or kill this script).
# A failure in steps 2-3 is logged and does not stop step 4. No analysis is run here.
set -u
P3="${P3_ROOT:-/root/autodl-tmp/paper3}"
PY="${PY:-python}"
POLL="${POLL:-60}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/usr/bin/shutdown}"
QLOG="${QLOG:-$P3/queue_g7_20261106.log}"
REPO="$P3/polyp-detection-reproducibility-protocol"
EV="$P3/evaluation/cache"
MS="$REPO/evaluation/multiseed_eval.py"; CF="$REPO/evaluation/crossover_fp_analysis.py"
EXP_MS=3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559
EXP_CF=c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2
SEEDS="${SEEDS:-50 51 52 53 54 55 56 57}"
log() { echo "[$(date '+%F %T')] $*"; }

log "等待 G7 训练结束（queue_g7 和 train_paper3.py 都不在）"
t0=$(date +%s); last=0
while :; do
  q=$(pgrep -f "^[^ ]*bash .*queue_g7[^ ]*\.sh" > /dev/null && echo 1 || echo 0)
  tr=$(pgrep -f "^[^ ]*python[0-9.]* .*train_paper3\.py( |$)" > /dev/null && echo 1 || echo 0)
  if [ "$q$tr" = 00 ]; then break; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 1800 ]; then log "仍在训练（已等 $(( (now - t0) / 60 )) 分钟）；队列最后一行：$(grep -a -E '^\[20' "$QLOG" 2>/dev/null | tail -1)"; last=$now; fi
  sleep "$POLL"
done
log "训练进程都已结束。队列日志最后一行：$(grep -a -E '^\[20' "$QLOG" 2>/dev/null | tail -1)"
sleep "${SETTLE:-30}"

OK=0
if grep -a -q "G7 队列全部结束" "$QLOG" 2>/dev/null; then OK=1; else log "队列日志里没有「G7 队列全部结束」：不缓存、不核对，直接进入关机步骤"; fi

# ---------- 2. one-time caches ----------
C7=0
if [ "$OK" = 1 ]; then
  DOI=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['addendum_doi'])" "$EV/ldpolyp/test/runs_g3__baseline640_seed42.json" 2>/dev/null)
  RUNS=""; n=0
  for s in $SEEDS; do for z in 640 1280; do d="$P3/runs_g7/baseline${z}_seed$s"
    if [ -f "$d/recovery_metadata.json" ]; then RUNS="$RUNS $d"; n=$((n+1)); else log "跳过未完成的 $d"; fi; done; done
  if [ -z "$DOI" ]; then log "读不到附录 DOI（G3 缓存记录），跳过缓存和核对"
  elif [ "$n" = 0 ]; then log "没有完成的 G7 run"
  else
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU 状态: /'
    log "缓存 G7：$n 个模型 × test、val（LDPolypVideo，各用训练时的输入尺寸）"
    if "$PY" "$P3/tools/cache_predictions.py" --run-dirs $RUNS --dataset "$P3/data/ldpolyp" --splits test val --out "$EV/ldpolyp" --doi "$DOI"; then
      C7=1; log "G7 缓存完成"; else log "G7 缓存出错（见上面），继续"; fi
    log "缓存文件数：ldpolyp test $(ls "$EV"/ldpolyp/test/runs_g7__*.npz 2>/dev/null | wc -l) 个 G7 npz（共 $(ls "$EV"/ldpolyp/test/*.npz 2>/dev/null | wc -l)），val $(ls "$EV"/ldpolyp/val/runs_g7__*.npz 2>/dev/null | wc -l) 个 G7 npz"
  fi
fi

# ---------- 3. end-to-end check against the public scripts (verification only) ----------
STAMP=$(date +%Y%m%d_%H%M%S)
check_size() {  # label imgsz seeds_csv baseline_csv candidate_csv
  local label=$1 sz=$2 seeds=$3 B=$4 C=$5 ds="$P3/data/ldpolyp" thr="$P3/data/size_thresholds.npy" cache="$EV/ldpolyp"
  local pub="$P3/evaluation/public_check_g7/${STAMP}_$label"
  mkdir -p "$pub"
  "$PY" - "$pub/jobs.json" "$P3" "$ds" "$thr" "$MS" "$CF" "$seeds" "$B" "$C" "$sz" <<'PYEOF' || { log "  $label：写 jobs.json 出错"; return 1; }
import hashlib, json, sys
import numpy as np
out, p3, ds, thr, ms, cf, seeds, b, c, sz = sys.argv[1:11]
seeds, b, c = seeds.split(","), b.split(","), c.split(",")
t = np.load(thr, allow_pickle=True).item()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
common = ["--dataset", ds, "--weights-dir", p3, "--imgsz", sz, "--conf", "0.25", "--iou-match", "0.5"]
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
    log "  $label：原样运行公开脚本 $j（真实推理，$sz）"
    ( cd "$pub" && eval "\"$PY\" $args" ) > "$pub/$j.log" 2>&1 || { log "  $label：$j 出错，见 $pub/$j.log"; return 1; }
  done
  "$PY" "$P3/tools/check_public_vs_cache.py" --repo "$REPO" --cache "$cache" --split test --jobs "$pub/jobs.json" --out "$pub/replay" > "$pub/check.log" 2>&1
  local rc=$?
  grep -E "identical|DIFFERENT|RESULT|exactly 0.25|images requested" "$pub/check.log" | sed "s/^/    /"
  log "  $label：核对退出码 $rc，结果目录 $pub"
  return $rc
}
if [ "$C7" = 1 ]; then
  if [ "$(sha256sum "$MS" | cut -c1-64)" = "$EXP_MS" ] && [ "$(sha256sum "$CF" | cut -c1-64)" = "$EXP_CF" ]; then
    log "端到端核对（S1 第 5 节，只作核对）：公开脚本哈希正确"
    SA=($SEEDS)
    for z in 640 1280; do
      Bz=""; Cz=""; Sz=""; miss=0
      for i in 0 1 2 3; do
        a=${SA[$i]}; b=${SA[$((i+4))]}
        fa="$P3/runs_g7/baseline${z}_seed$a"; fb="$P3/runs_g7/baseline${z}_seed$b"
        if [ -f "$fa/recovery_metadata.json" ] && [ -f "$fb/recovery_metadata.json" ]; then
          Bz="$Bz,runs_g7/baseline${z}_seed$a/weights/best.pt"; Cz="$Cz,runs_g7/baseline${z}_seed$b/weights/best.pt"; Sz="$Sz,${a}v$b"
        else miss=$((miss+1)); fi
      done
      if [ "$miss" -gt 0 ]; then log "$z：有 $miss 组种子没有完成，按 S1 只核对完整的组"; fi
      if [ -n "$Sz" ]; then
        check_size "g7_$z" "$z" "${Sz#,}" "${Bz#,}" "${Cz#,}" \
          && log "G7（$z）：缓存与公开脚本完全一致" || log "G7（$z）：核对没通过，先排查，暂停相关分析"
      else log "G7（$z）：没有可核对的组"; fi
    done
  else log "公开脚本哈希不对，跳过核对"; fi
else log "G7 缓存没有完成，跳过核对"; fi

# ---------- 4. shut down ----------
df -h "$P3" | tail -1 | sed 's/^/  磁盘: /'
sync
if [ -f "$P3/NO_SHUTDOWN" ]; then log "检测到 $P3/NO_SHUTDOWN，不关机"; exit 0; fi
log "关机：$SHUTDOWN_CMD"
sync
$SHUTDOWN_CMD
