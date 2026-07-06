# To use:
# bash watch_run.sh ./checkpoints/adjusting_bottle/finetunes/R2_noLSS_move_can_pot_seed42/log.txt

LOG="${1:-}"
N="${2:-8}"
[ -z "$LOG" ] && { echo "usage: bash watch_run.sh <logfile> [N]"; exit 1; }

# ---- one-time parse from the log header: mode, task, backbone, output dir ----
read_meta() {
    PS=$(pgrep -af "main.py" 2>/dev/null | head -1)
    # --- from log (reliable exact strings) ---
    MODE=$(grep -oE 'initialized in (finetune|pretrain) mode' "$LOG" 2>/dev/null | grep -oE 'finetune|pretrain' | head -1)
    TASK=$(grep -oE 'Single task [a-z0-9_]+' "$LOG" 2>/dev/null | head -1 | awk '{print $3}')
    EP=$(grep -oE 'Total [0-9]+ episodes' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    BB=$(grep -oE 'pretrained backbone from \./checkpoints/[A-Za-z0-9_.-]+' "$LOG" 2>/dev/null | head -1 | sed 's|.*/checkpoints/||')
    [ -z "$BB" ] && BB=$(grep -oE 'Loading pretrained backbone from \./checkpoints/[A-Za-z0-9_.-]+' "$LOG" 2>/dev/null | head -1 | sed 's|.*/checkpoints/||')
    NBB=$(grep -oE 'Loaded backbone with [0-9]+ parameters' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    OUT=$(grep -oE "checkpoints/[A-Za-z0-9_./-]+' already exists" "$LOG" 2>/dev/null | head -1 | sed "s|checkpoints/||;s|' already exists||;s|\./||")
    [ -z "$OUT" ] && OUT=$(echo "$PS" | grep -oE -- '--output_dir[ =]"?[^ ]+' | sed 's|.*=||;s|"||g;s|\./||;s|checkpoints/||' | head -1)
    CKPT="$HOME/H_RDT/checkpoints/$OUT"
    # LSS detection
    LSA="off"
    echo "$PS" | grep -qE -- '--use_dense_lsa' && LSA="DENSE"
    echo "$PS" | grep -qE -- '--use_lsa'       && LSA="pooled"
    echo "$PS" | grep -qE -- '--use_reasoning_head' && LSA="genhead(λ=$(echo "$PS" | grep -oE -- '--reasoning_lambda[ =][0-9.]+' | grep -oE '[0-9.]+' | head -1))"
    [ "$MODE" = "finetune" ] && LSA="off (finetune)"
}
read_meta

while true; do
    clear
    pgrep -f "main.py" >/dev/null && ST=$'\033[32mRUNNING\033[0m' || ST=$'\033[31mSTOPPED\033[0m'

    # latest tqdm progress line (handle \r bars)
    PROG=$(tail -c 60000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -E 'Steps:|it/s' | tail -1)
    CUR=$(echo "$PROG"  | grep -oE '[0-9]+/[0-9]+' | head -1)
    PCT=$(echo "$PROG"  | grep -oE '[0-9]+%' | head -1)
    ETA=$(echo "$PROG"  | grep -oE '<[0-9:]+' | head -1 | tr -d '<')
    ELAP=$(echo "$PROG" | grep -oE '\[[0-9:]+<' | head -1 | tr -d '[<')
    RATE=$(echo "$PROG" | grep -oE '[0-9.]+s/it' | head -1)
    LOSS=$(echo "$PROG" | grep -oE 'loss=[0-9.e-]+' | head -1)
    DIFF=$(echo "$PROG" | grep -oE 'diff_loss=[0-9.e-]+' | head -1)
    RSN=$(echo "$PROG" | grep -oE 'reasoning_loss=[0-9.e-]+' | head -1)
    LR=$(echo "$PROG"   | grep -oE 'lr=[0-9.e-]+' | head -1)

    echo -e "===== H-RDT MONITOR  $(date '+%H:%M:%S')   status: $ST ====="
    echo "run     : mode=${MODE:-?}  task=${TASK:-?}  episodes=${EP:-?}  LSS=${LSA}"
    echo "backbone: ${BB:-?}  (${NBB:-?} params loaded)"
    echo "outdir  : $(basename "${CKPT:-?}")"
    echo "------------------------------------------------------------"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{printf "GPU%s: util %3s%%  mem %5s/%5sMiB  %sC  %sW\n",$1,$2,$3,$4,$5,$6}'
    echo "------------------------------------------------------------"
    echo "progress: ${CUR:-?}  (${PCT:-?})   elapsed ${ELAP:-?}  ETA ${ETA:-?}  @ ${RATE:-?}"
    echo "loss    : ${LOSS:-?}   ${DIFF:-}   ${RSN:-}   ${LR:-}"
    CKPTS=$(ls -1d "$CKPT"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tr '\n' ' ')
    echo "ckpts   : ${CKPTS:-none yet}"
    [ -f "$CKPT/pytorch_model.bin" ] && echo "FINAL   : pytorch_model.bin present (run complete)"
    echo "---------------- last $N progress lines --------------------"
    tail -c 30000 "$LOG" 2>/dev/null | tr '\r' '\n' \
        | grep -vE "torch.Size|^[[:space:]]*$" | tail -n "$N"
    echo "------------------------------------------------------------"
    echo "(refresh 3s — Ctrl-C stops watching, not training)"
    # re-read meta occasionally in case the run just started writing header lines
    [ -z "$NBB" ] && read_meta
    sleep 3
done
