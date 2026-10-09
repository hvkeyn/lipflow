#! /bin/bash
# Fine-tune MuAViC's Russian audio-visual model into a LIP-READING model: their published
# train.sh with the audio modality removed, plus single-GPU (RTX 3060 12 GB) ergonomics.
#
# Read RUNBOOK.md first - especially the timeline and the "measure the baseline" step.
#
# Fill in the paths below, then: bash train_video_only.sh
set -euo pipefail

LANG=ru
OUT_PATH=/path/to/out
PRETRAINED_MODEL_PATH=/path/to/muavic/models/ru_avsr/checkpoint_best.pt

# Same layout as MuAViC's README: muavic/av_hubert and muavic/muavic side by side.
ROOT=/path/to/muavic
AV_HUBERT=${ROOT}/av_hubert/avhubert
MUAVIC=${ROOT}/muavic

TGT=${LANG}
DATA_PATH=${MUAVIC}/${LANG}

# --- labels: SentencePiece + fairseq dict, built exactly as MuAViC builds them -------------
LABEL_DIR=${OUT_PATH}/${LANG}/labels
mkdir -p "${LABEL_DIR}"
TOKENIZER_PATH=${LABEL_DIR}/tokenizer.model
if [[ ! -f ${TOKENIZER_PATH} ]]; then
    spm_train \
        --input="${DATA_PATH}/train.${TGT}" \
        --model_prefix="${LABEL_DIR}/tokenizer" \
        --vocab_size=1000 \
        --model_type=unigram \
        --character_coverage=1.0 \
        --bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3
    awk 'FNR > 4 {print $(NF-1)" "1;}' "${LABEL_DIR}/tokenizer.vocab" > "${LABEL_DIR}/dict.${TGT}.txt"
    cp "${DATA_PATH}"/{train,valid,test}."${TGT}" "${LABEL_DIR}/"
fi

export PYTHONPATH="${ROOT}/av_hubert/fairseq:${PYTHONPATH:-}"

# --- changed from MuAViC's train.sh --------------------------------------------------------
#   task.modalities=[video]           the whole point: audio is never fed in, so the decoder
#                                     must learn to use the visual stream it currently ignores
#   dataset.batch_size=2              one 12 GB card; the encoder is 387 M parameters
#   optimization.update_freq=[16]     gradient accumulation: an effective batch of 32
#   distributed_training.*            single GPU (their recipe assumes 8 GPUs x 4 nodes)
#   optimization.max_update=300000    their 30000 assumes 32 GPUs; see RUNBOOK.md
#   checkpoint.save_interval_updates  so weeks of training can be stopped without losing it
#   model.freeze_finetune_updates     longer: the decoder needs more time than the encoder
#   common.fp16=true                  AMP; the key may be common.bf16=true on Ampere instead
#
# Verify every key against ${AV_HUBERT}/conf/finetune/base_vox_433h.yaml in your checkout -
# that file is the source of truth, and these scripts have not been run (see RUNBOOK.md).
fairseq-hydra-train \
    --config-dir "${AV_HUBERT}/conf/finetune" \
    --config-name base_vox_433h \
        common.user_dir="${AV_HUBERT}" \
        common.fp16=true \
        task.data="${DATA_PATH}" \
        task.modalities=[video] \
        task.label_dir="${LABEL_DIR}" \
        task.labels=["${TGT}"] \
        +task.min_sample_size=5 \
        task.max_sample_size=500 \
        task.tokenizer_bpe_model="${TOKENIZER_PATH}" \
        dataset.train_subset=train \
        dataset.valid_subset=valid \
        dataset.batch_size=2 \
        dataset.num_workers=4 \
        optimization.update_freq=[16] \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        model.w2v_path="${PRETRAINED_MODEL_PATH}" \
        model.freeze_finetune_updates=10000 \
        model.dropout=0.1 \
        +model.no_pretrained_weights=false \
        optimization.lr=[0.0001] \
        optimization.max_update=300000 \
        lr_scheduler.warmup_steps=20000 \
        lr_scheduler.decay_steps=250000 \
        checkpoint.save_interval_updates=2000 \
        checkpoint.keep_interval_updates=3 \
        hydra.run.dir="${OUT_PATH}/${LANG}"

# When this finishes (or when you stop it), measure it with decode_video.sh pointed at
# ${OUT_PATH}/${LANG}/checkpoint_best.pt and compare the WER with the baseline you took first.
