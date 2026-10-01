#!/bin/bash
# Benchmarks mixel on this Mac and writes a shareable report,
# mixel-benchmark-<chip>-<gpu>-<memory>-<date>.md, labelled with the hardware.
#
#   scripts/benchmark.sh              # run everything that fits in memory
#   scripts/benchmark.sh --dry-run    # only show the plan
#
# Options:
#   --dry-run      Show what would run, and what it would download, then stop.
#   --yes          Don't ask before downloading model weights.
#   --force        Also run tests that may not fit in memory (expect swapping).
#   --mixel PATH   Benchmark this binary instead of building the repo's.
#   --out DIR      Write the report here (default: the current directory).
#   --keep         Keep the generated images and run logs.
#
# Please email the report to mixelate@proton.me so we can compile results
# across Macs. It holds the hardware summary and timings only: no serial
# numbers, hostnames, user names, or images.
set -u

REPORT_EMAIL="mixelate@proton.me"
PROMPT="A red fox sitting in fresh snow, wildlife photography"
EDIT_PROMPT="Turn the fox into a gray wolf"
SEED=1
REF_HW="Apple M3 Max, 30-core GPU, 96 GB"
GIB=1073741824

# id | label | model | extra args | steps | runs | peak GB | reference s | reference s/step
# Peak memory and reference times are from $REF_HW.
TESTS=(
  "z512|Z-Image-Turbo, 512x512|z-image-turbo|--width 512 --height 512|9|3|26.7|13.2|1.4"
  "z1024|Z-Image-Turbo, 1024x1024|z-image-turbo||9|3|36.4|59.0|6.4"
  "fast|Qwen-Image-2.1 fast, 1024x1024|qwen-image-2.1-fast||4|3|52.1|36.8|8.7"
  "fastedit|Qwen-Image-2.1 fast, edit|qwen-image-2.1-fast|--ref-image REF|4|3|64.8|63.6|11.6"
  "qwen|Qwen-Image-2.1, 1024x1024|qwen-image-2.1||40|3|51.5|373.0|9.3"
  "qwenedit|Qwen-Image-2.1, edit|qwen-image-2.1|--ref-image REF|40|3|64.2|444.5|10.7"
)
# Models (Hugging Face repo, download GB) in the order the tests use them.
repo_of() {
  case "$1" in
    z-image-turbo) echo "Tongyi-MAI/Z-Image-Turbo" ;;
    qwen-image-2.1) echo "Qwen/Qwen-Image-2.1" ;;
    qwen-image-2.1-fast) echo "Qwen/Qwen-Image-2.1 alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs" ;;
  esac
}
download_gb() {
  case "$1" in
    Tongyi-MAI/Z-Image-Turbo) echo 33 ;;
    Qwen/Qwen-Image-2.1) echo 31 ;;
    alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) echo 0.4 ;;
  esac
}

DRY_RUN=0 YES=0 FORCE=0 KEEP=0 MIXEL="" OUT="."
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes) YES=1 ;;
    --force) FORCE=1 ;;
    --keep) KEEP=1 ;;
    --mixel) MIXEL="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
num() { awk "BEGIN { printf \"%.1f\", $1 }"; }

# ---------- hardware (nothing that identifies the machine or its owner) ----------
CHIP="$(sysctl -n machdep.cpu.brand_string)"
MODEL_ID="$(sysctl -n hw.model)"
MODEL_NAME="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Model Name/ {print $2; exit}')"
MEM_BYTES="$(sysctl -n hw.memsize)"
MEM_GB="$(awk "BEGIN { printf \"%d\", $MEM_BYTES / $GIB }")"
P_CORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 0)"
E_CORES="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)"
GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ {print $2; exit}')"
OS_VERSION="$(sw_vers -productVersion)"
OS_BUILD="$(sw_vers -buildVersion)"
case "$(pmset -g ps | head -1)" in
  *"AC Power"*) POWER="AC power" ;;
  *"Battery Power"*) POWER="battery" ;;
  *) POWER="unknown" ;;
esac
case "$(pmset -g | awk '$1 == "powermode" {print $2}')" in
  0) ENERGY="Automatic" ;;
  1) ENERGY="Low Power" ;;
  2) ENERGY="High Power" ;;
  *) ENERGY="n/a" ;;
esac
HW_LABEL="$CHIP, ${GPU_CORES:-?}-core GPU, ${MEM_GB} GB"

# ---------- plan: what fits, what downloads ----------
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
cached() { [ -d "$HF_HUB/models--$(echo "$1" | sed 's|/|--|')" ]; }
LIMIT_GB="$(awk "BEGIN { printf \"%.1f\", $MEM_GB * 0.85 }")"
fits() { awk "BEGIN { exit !($1 <= $LIMIT_GB) }"; }

RUN_IDS="" DOWNLOADS="" DOWNLOAD_GB=0
echo "mixel benchmark on: $HW_LABEL ($MODEL_NAME, macOS $OS_VERSION, $POWER)"
echo "Tests use up to 85% of memory ($LIMIT_GB GB here); larger ones are skipped."
echo
for t in "${TESTS[@]}"; do
  IFS='|' read -r id label model extra steps runs peak ref ref_step <<<"$t"
  if fits "$peak" || [ $FORCE -eq 1 ]; then
    RUN_IDS="$RUN_IDS $id"
    printf "  run   %-34s needs ~%s GB\n" "$label" "$peak"
    for r in $(repo_of "$model"); do
      if ! cached "$r" && ! echo "$DOWNLOADS" | grep -q " $r"; then
        DOWNLOADS="$DOWNLOADS $r"
        DOWNLOAD_GB="$(num "$DOWNLOAD_GB + $(download_gb "$r")")"
      fi
    done
  else
    printf "  skip  %-34s needs ~%s GB\n" "$label" "$peak"
  fi
done
echo
if [ -z "$RUN_IDS" ]; then
  echo "Nothing fits in $MEM_GB GB (the smallest test needs ~27 GB). --force tries anyway."
  exit 1
fi
if [ -n "$DOWNLOADS" ]; then
  echo "Will download ~$DOWNLOAD_GB GB of model weights to $HF_HUB:"
  for r in $DOWNLOADS; do echo "  $r (~$(download_gb "$r") GB)"; done
else
  echo "All needed weights are already downloaded."
fi
EST_MIN=0
for t in "${TESTS[@]}"; do
  IFS='|' read -r id label model extra steps runs peak ref ref_step <<<"$t"
  case " $RUN_IDS " in *" $id "*) EST_MIN="$(num "$EST_MIN + $runs * ($ref + 5) / 60")" ;; esac
done
echo "Each test runs 3 times. On an $REF_HW this takes ~$(printf "%.0f" "$EST_MIN") min; slower chips take longer."
[ $DRY_RUN -eq 1 ] && exit 0
if [ -n "$DOWNLOADS" ] && [ $YES -eq 0 ]; then
  if [ -t 0 ]; then
    printf "Continue? [y/N] "
    read -r answer
    case "$answer" in y|Y|yes) ;; *) echo "Stopped."; exit 1 ;; esac
  else
    echo "Downloads needed; rerun with --yes to allow them." >&2
    exit 1
  fi
fi
echo

# ---------- the binary ----------
if [ -z "$MIXEL" ]; then
  echo "Building mixel (cargo build --release)..."
  (cd "$REPO_DIR" && cargo build --release --quiet) || { echo "build failed" >&2; exit 1; }
  MIXEL="$REPO_DIR/target/release/mixel"
fi
MIXEL_VERSION="$("$MIXEL" --version 2>/dev/null)"
GIT_REV="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  GIT_REV="$GIT_REV (modified)"
fi
MLX_RS="$(awk '/^name = "mlx-rs"/ {getline; gsub(/"/, "", $3); print $3}' "$REPO_DIR/Cargo.lock" 2>/dev/null)"

WORK="$(mktemp -d -t mixel-benchmark)"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_S=$(date +%s)
swap_used_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i < NF; i++) if ($i == "used") {v = $(i + 2); sub(/M/, "", v); print v}}'; }

# One mixel run; sets R_* from its output and /usr/bin/time.
run_mixel() {
  local log="$1"; shift
  local swap0 swap1
  swap0="$(swap_used_mb)"
  /usr/bin/time -l "$MIXEL" "$@" >"$log" 2>&1
  R_STATUS=$?
  swap1="$(swap_used_mb)"
  R_LOAD="$(sed -n 's/^Loaded in \([0-9.]*\)s.*/\1/p' "$log")"
  local timings
  timings="$(grep '^Timings:' "$log")"
  R_TEXT="$(echo "$timings" | sed -n 's/.*text \([0-9.]*\)s.*/\1/p')"
  R_INIT="$(echo "$timings" | sed -n 's/.*init image \([0-9.]*\)s.*/\1/p')"
  R_DENOISE="$(echo "$timings" | sed -n 's/.*denoise \([0-9.]*\)s.*/\1/p')"
  R_STEP="$(echo "$timings" | sed -n 's/.*(\([0-9.]*\)s\/step).*/\1/p')"
  R_VAE="$(echo "$timings" | sed -n 's/.*VAE \([0-9.]*\)s.*/\1/p')"
  R_IMAGE="$(sed -n 's/^Done! Image saved to .* (\([0-9.]*\)s)$/\1/p' "$log")"
  R_PEAK="$(awk -v g=$GIB '/peak memory footprint/ {printf "%.1f", $1 / g}' "$log")"
  R_SWAP="$(awk "BEGIN { d = ($swap1 - $swap0) / 1024; printf \"%.1f\", (d > 0 ? d : 0) }")"
  R_ERROR="$(grep -E '^Error' "$log" | tail -1 | sed 's/"/'"'"'/g')"
  if [ $R_STATUS -ne 0 ] || [ -z "$R_IMAGE" ]; then R_STATUS=1; fi
}

TABLE="" REF_IMAGE="" WARMED=""
JSON_ITEMS=()
for t in "${TESTS[@]}"; do
  IFS='|' read -r id label model extra steps runs peak ref ref_step <<<"$t"
  case " $RUN_IDS " in *" $id "*) ;; *)
    TABLE="$TABLE| $label | skipped: needs ~$peak GB | | | | $ref s |\n"
    JSON_ITEMS+=("{\"test\": \"$id\", \"label\": \"$label\", \"model\": \"$model\", \"status\": \"skipped\", \"needs_gb\": $peak}")
    continue ;;
  esac
  args=(--model "$model" --seed "$SEED" --output "$WORK/$id.png")
  case "$extra" in
    *REF*)
      if [ -z "$REF_IMAGE" ]; then
        TABLE="$TABLE| $label | skipped: no reference image | | | | $ref s |\n"
        JSON_ITEMS+=("{\"test\": \"$id\", \"label\": \"$label\", \"model\": \"$model\", \"status\": \"skipped\", \"reason\": \"no reference image\"}")
        continue
      fi
      args+=(--prompt "$EDIT_PROMPT" ${extra//REF/$REF_IMAGE}) ;;
    *) args+=(--prompt "$PROMPT" $extra) ;;
  esac
  # A 1-step warm-up per model: downloads the weights on first use and
  # pages them into memory, so the timed runs measure generation only.
  if ! echo "$WARMED " | grep -q " $model "; then
    echo "Warming up $model (downloads its weights the first time)..."
    run_mixel "$WORK/warmup-$model.log" --model "$model" --prompt "$PROMPT" --seed "$SEED" \
      --width 512 --height 512 --num-steps 1 --output "$WORK/warmup-$model.png"
    WARMED="$WARMED $model"
  fi
  ok_runs="" runs_json="" max_peak=0 max_swap=0
  for n in $(seq 1 "$runs"); do
    echo "$label, run $n of $runs..."
    run_mixel "$WORK/$id-$n.log" "${args[@]}"
    if [ $R_STATUS -ne 0 ]; then
      echo "  failed: ${R_ERROR:-see $WORK/$id-$n.log}"
      runs_json="$runs_json{\"status\": \"failed\", \"error\": \"$R_ERROR\"}, "
      KEEP=1
      continue
    fi
    echo "  ${R_IMAGE} s (${R_STEP} s/step), peak ${R_PEAK} GB"
    runs_json="$runs_json{\"load_s\": ${R_LOAD:-null}, \"text_s\": $R_TEXT, \"init_image_s\": $R_INIT, \"denoise_s\": $R_DENOISE, \"per_step_s\": $R_STEP, \"vae_s\": $R_VAE, \"image_s\": $R_IMAGE, \"peak_gb\": ${R_PEAK:-null}, \"swap_gb\": $R_SWAP}, "
    ok_runs="$ok_runs$R_IMAGE $R_STEP"$'\n'
    max_peak="$(awk "BEGIN { print ($R_PEAK > $max_peak ? $R_PEAK : $max_peak) }")"
    max_swap="$(awk "BEGIN { print ($R_SWAP > $max_swap ? $R_SWAP : $max_swap) }")"
  done
  [ "$id" = fast ] && [ -f "$WORK/fast.png" ] && REF_IMAGE="$WORK/fast.png"
  runs_json="[${runs_json%, }]"
  if [ -z "$ok_runs" ]; then
    TABLE="$TABLE| $label | failed | | | | $ref s |\n"
    status=failed
  else
    # Median run (by time per image), and the range over all runs.
    sorted="$(printf "%s" "$ok_runs" | sort -n)"
    count="$(printf "%s\n" "$sorted" | wc -l | tr -d ' ')"
    read -r med med_step <<<"$(printf "%s\n" "$sorted" | sed -n "$(((count + 1) / 2))p")"
    lo="$(printf "%s\n" "$sorted" | head -1 | cut -d' ' -f1)"
    hi="$(printf "%s\n" "$sorted" | tail -1 | cut -d' ' -f1)"
    note=""
    [ "$count" -lt "$runs" ] && note=" ($count of $runs runs)"
    awk "BEGIN { exit !($max_swap > 0.5) }" && note="$note (swapped ${max_swap} GB)"
    TABLE="$TABLE| $label | **$med s**$note | $lo-$hi s | $med_step s | $max_peak GB | $ref s |\n"
    status=ok
  fi
  JSON_ITEMS+=("{\"test\": \"$id\", \"label\": \"$label\", \"model\": \"$model\", \"steps\": $steps, \"status\": \"$status\", \"reference_image_s\": $ref, \"runs\": $runs_json}")
done
MINUTES="$(num "($(date +%s) - $START_S) / 60")"

# ---------- the report ----------
SAFE_LABEL="$(echo "$CHIP-${GPU_CORES:-x}gpu-${MEM_GB}GB" | tr ' ' '-')"
REPORT="$OUT/mixel-benchmark-$SAFE_LABEL-$(date +%Y-%m-%d).md"
{
  echo "# mixel benchmark: $HW_LABEL"
  echo
  echo "> **Please email this file to $REPORT_EMAIL** so we can compile results across Macs."
  echo "> It holds the hardware summary and timings below, nothing else: no serial numbers,"
  echo "> hostnames, user names or images."
  echo
  echo "## Hardware"
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| Mac | ${MODEL_NAME:-unknown} ($MODEL_ID) |"
  echo "| Chip | $CHIP |"
  echo "| CPU | $((P_CORES + E_CORES)) cores ($P_CORES performance, $E_CORES efficiency) |"
  echo "| GPU | ${GPU_CORES:-?} cores |"
  echo "| Memory | $MEM_GB GB |"
  echo "| macOS | $OS_VERSION ($OS_BUILD) |"
  echo "| Power | $POWER, energy mode $ENERGY |"
  echo
  echo "## Software"
  echo
  echo "$MIXEL_VERSION (git $GIT_REV), mlx-rs ${MLX_RS:-?}. Run $STARTED, took $MINUTES min."
  echo
  echo "## Results"
  echo
  echo "| Test | Time per image (median) | Range | Per step | Peak memory | Reference* |"
  echo "|---|---:|---:|---:|---:|---:|"
  printf "%b" "$TABLE"
  echo
  echo "\\* The same test on an $REF_HW, for comparison."
  echo
  echo "## How to read this"
  echo
  echo "- **Time per image**: generating one image with the model already loaded (text encoding,"
  echo "  denoising, VAE decoding): the median of 3 runs back to back, with no pause, as in a"
  echo "  batch. **Range** is the fastest and slowest of them: a Mac slows down as it heats up,"
  echo "  so a wide range mostly shows how much this one throttles under sustained load. Loading"
  echo "  the model adds a few seconds per run of mixel; the first run of a model also downloads it."
  echo "- **Per step**: denoising time per step, the part that grows with the step count: Z-Image-Turbo"
  echo "  runs 9 steps, Qwen-Image-2.1 fast 4 and Qwen-Image-2.1 40."
  echo "- **Peak memory**: the most memory mixel used (1 GB = 2^30 bytes, as Apple counts RAM)."
  echo "  Tests that need more than 85% of this Mac's memory are skipped. \"Swapped\" means macOS"
  echo "  moved memory to disk during the test, which makes it slower than the chip can do."
  echo "- **Settings**: prompt \"$PROMPT\" (edits: \"$EDIT_PROMPT\","
  echo "  on the fast test's image), seed $SEED. The tests run in the order above, so the later"
  echo "  ones start on an already warm Mac."
  echo
  echo "## Raw data"
  echo
  echo '```json'
  echo "{"
  echo "  \"schema\": 1,"
  echo "  \"hardware\": {\"mac\": \"${MODEL_NAME:-unknown}\", \"model_id\": \"$MODEL_ID\", \"chip\": \"$CHIP\", \"cpu_performance_cores\": $P_CORES, \"cpu_efficiency_cores\": $E_CORES, \"gpu_cores\": ${GPU_CORES:-null}, \"memory_gb\": $MEM_GB, \"macos\": \"$OS_VERSION\", \"macos_build\": \"$OS_BUILD\", \"power\": \"$POWER\", \"energy_mode\": \"$ENERGY\"},"
  echo "  \"software\": {\"mixel\": \"$MIXEL_VERSION\", \"git\": \"$GIT_REV\", \"mlx_rs\": \"${MLX_RS:-}\"},"
  echo "  \"run\": {\"started\": \"$STARTED\", \"minutes\": $MINUTES, \"prompt\": \"$PROMPT\", \"edit_prompt\": \"$EDIT_PROMPT\", \"seed\": $SEED},"
  echo "  \"results\": ["
  for i in "${!JSON_ITEMS[@]}"; do
    sep=","
    [ "$i" -eq $((${#JSON_ITEMS[@]} - 1)) ] && sep=""
    echo "    ${JSON_ITEMS[$i]}$sep"
  done
  echo "  ]"
  echo "}"
  echo '```'
} >"$REPORT"

if [ $KEEP -eq 1 ]; then echo "Logs and images kept in $WORK"; else rm -rf "$WORK"; fi
echo
echo "Report: $REPORT"
echo "Please email it to $REPORT_EMAIL. Thank you!"
