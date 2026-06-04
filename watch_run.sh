# To use:
# bash ~/H_RDT/watch_run.sh ~/H_RDT/logs_XXX.txt

LOG="${1:-}"
N="${2:-8}"
[ -z "$LOG" ] && { echo "usage: bash watch_run.sh <logfile> [N]"; exit 1; }

# ---- one-time parse from the log header: mode, task, backbone, output dir ----
read_meta() {
    MODE=$(grep -oE 'mode[ =]+"?(pretrain|finetune)' "$LOG" 2>/dev/null | grep -oE 'pretrain|finetune' | head -1)
    [ -z "$MODE" ] && MODE=$(grep -qiE 'finetune mode with pretrained backbone' "$LOG" 2>/dev/null && echo finetune || echo "")
    [ -z "$MODE" ] && MODE=$(grep -qiE 'LSAHead initialized|Constructing model from pretrained' "$LOG" 2>/dev/null && echo pretrain || echo "?")
    TASK=$(grep -oE 'Single task [a-z0-9_]+' "$LOG" 2>/dev/null | head -1 | awk '{print $3}')
    [ -z "$TASK" ] && TASK=$(grep -oE -- '--task_name[ =]"?[a-z0-9_]+' "$LOG" 2>/dev/null | head -1 | grep -oE '[a-z0-9_]+$')
    EPISODES=$(grep -oE 'Total [0-9]+ episodes' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    BB=$(grep -oE 'pretrained backbone from .*pytorch_model.bin' "$LOG" 2>/dev/null | head -1 | sed 's|.*/checkpoints/||;s|/pytorch_model.bin||')
    [ -z "$BB" ] && BB=$(grep -oE -- '--pretrained_backbone_path[ =]"?[^ ]+' "$LOG" 2>/dev/null | head -1 | sed 's|.*/checkpoints/||;s|/pytorch_model.bin.*||')
    NBB=$(grep -oE 'Loaded backbone with [0-9]+ parameters' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    OUT=$(grep -oE -- '--output_dir[ =]"?[^ ]+' "$LOG" 2>/dev/null | head -1 | sed 's|.*=||;s|"||g')
    [ -z "$OUT" ] && OUT=$(grep -oE 'checkpoints/[A-Za-z0-9_./-]+' "$LOG" 2>/dev/null | grep -vE 'pretrain-0618|pretrain_human' | head -1)
    CKPT="$HOME/H_RDT/$OUT"
    [ ! -d "$CKPT" ] && CKPT=$(dirname "$(ls -dt $HOME/H_RDT/checkpoints/*/checkpoint-* 2>/dev/null | head -1)" 2>/dev/null)
    LSA=$(grep -qiE 'LSAHead initialized|use_lsa.*[Tt]rue|lsa_loss' "$LOG" 2>/dev/null && echo "ON" || echo "off")
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
    LR=$(echo "$PROG"   | grep -oE 'lr=[0-9.e-]+' | head -1)

    echo -e "===== H-RDT MONITOR  $(date '+%H:%M:%S')   status: $ST ====="
    echo "run     : mode=${MODE:-?}  task=${TASK:-?}  episodes=${EPISODES:-?}  LSS=${LSA}"
    echo "backbone: ${BB:-?}  (${NBB:-?} params loaded)"
    echo "outdir  : $(basename "${CKPT:-?}")"
    echo "------------------------------------------------------------"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{printf "GPU%s: util %3s%%  mem %5s/%5sMiB  %sC  %sW\n",$1,$2,$3,$4,$5,$6}'
    echo "------------------------------------------------------------"
    echo "progress: ${CUR:-?}  (${PCT:-?})   elapsed ${ELAP:-?}  ETA ${ETA:-?}  @ ${RATE:-?}"
    echo "loss    : ${LOSS:-?}   ${DIFF:-}   ${LR:-}"
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
