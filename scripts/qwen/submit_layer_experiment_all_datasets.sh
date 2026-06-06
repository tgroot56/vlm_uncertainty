#!/bin/bash

set -euo pipefail

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-qwen_layer_sweep_seed${SEED}}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ -z "${RUN_SUFFIX:-}" ]]; then
  if (( MAX_SAMPLES % 1000 == 0 )); then
    RUN_SUFFIX="_$((MAX_SAMPLES / 1000))k"
  else
    RUN_SUFFIX="_${MAX_SAMPLES}"
  fi
fi

export MAX_SAMPLES SEED RUN_NAME RUN_SUFFIX NUM_EPOCHS BATCH_SIZE

mkdir -p slurm_outputs/qwen

submit_dataset() {
  local script="$1"
  sbatch --parsable \
    --export=ALL,MAX_SAMPLES,RUN_NAME,RUN_SUFFIX,SEED,NUM_EPOCHS,BATCH_SIZE \
    "$script"
}

vqa_job="$(submit_dataset "$HOME/vlm_uncertainty/scripts/qwen/vqa-v2/layer_experiment.job")"
coco_job="$(submit_dataset "$HOME/vlm_uncertainty/scripts/qwen/coco-qa-vi/layer_experiment.job")"
imagenet_job="$(submit_dataset "$HOME/vlm_uncertainty/scripts/qwen/imagenet-r/layer_experiment.job")"
pope_job="$(submit_dataset "$HOME/vlm_uncertainty/scripts/qwen/pope/random/layer_experiment_random.job")"

dependency="afterok:${vqa_job}:${coco_job}:${imagenet_job}:${pope_job}"
plot_job="$(sbatch --parsable \
  --dependency="$dependency" \
  --export=ALL,RUN_SUFFIX \
  "$HOME/vlm_uncertainty/scripts/qwen/plot_layer_experiment_all_datasets.job")"

echo "Submitted Qwen layer-sweep jobs"
echo "  max samples: $MAX_SAMPLES"
echo "  run name:    $RUN_NAME"
echo "  run suffix:  $RUN_SUFFIX"
echo "  VQA-v2:      $vqa_job"
echo "  COCO-QA:     $coco_job"
echo "  ImageNet-R:  $imagenet_job"
echo "  POPE:        $pope_job"
echo "  plot job:    $plot_job ($dependency)"
