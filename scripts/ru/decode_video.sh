#! /bin/bash
# Measure the released MuAViC Russian checkpoint on the test set with VIDEO ONLY.
#
# This is step 3 of RUNBOOK.md and the cheapest useful thing you can run: no training, a few
# hours, and it tells you whether the visual stream carries any Russian at all before you spend
# weeks teaching the model to use it.
#
# Fill in the paths below, then: bash decode_video.sh
set -euo pipefail

LANG=ru
GROUP=test           # or valid
MODALITIES=video     # <- the point: no audio is fed in at all
MODEL_PATH=/path/to/muavic/models/ru_avsr/checkpoint_best.pt
TOKENIZER_PATH=/path/to/out/ru/labels/tokenizer.model    # built by train_video_only.sh step 1
OUT_PATH=/path/to/out

ROOT=/path/to/muavic
AV_HUBERT=${ROOT}/av_hubert/avhubert
MUAVIC=${ROOT}/muavic

TGT=${LANG}
DATA_PATH=${MUAVIC}/${LANG}

export PYTHONPATH="${ROOT}/av_hubert/fairseq:${PYTHONPATH:-}"

# MuAViC's decode.sh, with override.modalities=[video]. WER is printed at the end; compare it
# with the same command run twice - once with [audio] and once with [audio,video] - to see how
# much the video stream is actually contributing.
python -B "${AV_HUBERT}/infer_s2s.py" \
    --config-dir "${AV_HUBERT}/conf" \
    --config-name s2s_decode \
        common.user_dir="${AV_HUBERT}" \
        override.modalities=["${MODALITIES}"] \
        dataset.gen_subset="${GROUP}" \
        override.data="${DATA_PATH}" \
        override.label_dir="${DATA_PATH}" \
        override.tokenizer_bpe_model="${TOKENIZER_PATH}" \
        generation.beam=5 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.labels=["${TGT}"] \
        override.eval_bleu=false \
        common_eval.path="${MODEL_PATH}" \
        common_eval.results_path="${OUT_PATH}/${LANG}_avsr_${MODALITIES}"
