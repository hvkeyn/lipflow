# Russian lip reading: fine-tuning the MuAViC model to work from video alone

This is the path to actual **silent** Russian lip reading. Everything else in this repository's
Russian support (see `lipflow/asr.py`) is voice recognition; this directory is about training a
visual model, which is the only way to get Russian from lips.

Read the whole file before starting: step 3 alone may change your mind about steps 4–5.

## Why this is needed

* There is **no public Russian visual-only model**. The audit is in the session that produced
  this branch: Auto-AVSR ships English only; mpc001's multilingual VSR has no Russian; MuAViC
  released AVSR and AVST checkpoints only; XLAVS-R released no weights; Zero-AVSR's Russian
  weights are audio-visual with **no licence file**.
* The closest thing, MuAViC's `ru_avsr`, **does not read lips**: decoding the Russian example
  video-only, exactly as its own `MODALITIES=video` does it (audio features zeroed), gives
  incoherent Russian — 7 of the 28 words of the reference reading — while the same checkpoint
  with the real audio transcribes it correctly. The video stream is wired (the encoder state
  moves by 0.41 when it is replaced with zeros) but the fine-tuned decoder does not use it.
* The good news: that checkpoint **already knows Russian** — a 996-subword SentencePiece
  vocabulary and a decoder trained on 49 h of Russian audio-visual speech. What it lacks is the
  habit of reading lips. Forcing the audio modality off while fine-tuning is what teaches it.

## Hardware and honest expectations

Target here: **Windows + WSL2 + one RTX 3060 12 GB**.

| | |
|---|---|
| Model | AV-HuBERT large, 387 M parameters, 1.55 GB in fp32 (the fairseq checkpoint is 4.62 GB because it also carries optimizer state) |
| Fits in 12 GB? | Yes, but only at batch 1–2 with gradient accumulation |
| Data | MuAViC-ru: 49 h train / 238 speakers, 1.7 h valid, 1.8 h test. Tens of GB on disk |
| Licence | **CC-BY-NC 4.0**, non-commercial. Both the data and any model derived from it |
| Official recipe | MuAViC's `train.sh`: `optimization.max_update=30000`, `model.freeze_finetune_updates=4000`, run on 8 GPUs × 4 nodes |

**The timeline is weeks, not hours.** The official recipe was tuned for 32 GPUs; 30 000 updates
on one 3060 see about 1/32 of the data, so reaching a comparable number of samples needs tens of
times more updates. Plan for **2–4 weeks of continuous training** for a real run, and treat any
figure below that as a subset experiment with correspondingly worse quality.

**Quality expectations.** Visual-only is much harder than audio-visual: on English, AV-HuBERT
scores 0.9–1.7 % WER with audio and 19.1 % video-only. MuAViC-ru gets 30.0 % audio-visual, and
visual-only for Russian has never been published. Published figures for low-resource visual
speech recognition are 44–59 % WER at ~20 h. Expect a model that reads *some* words and is often
wrong — usable as an experiment, not as the experience the English mode gives you.

## Steps

### 0. WSL2 with CUDA

```powershell
wsl --install -d Ubuntu-22.04
```
Then inside WSL, install the NVIDIA driver for WSL (from Windows, not inside WSL), CUDA toolkit
12.1, miniconda, and `ffmpeg`/`sox`. Verify with `nvidia-smi` **inside WSL** before going on —
fairseq will not build on Windows itself.

### 1. The MuAViC + AV-HuBERT stack

```bash
git clone https://github.com/facebookresearch/muavic.git
cd muavic
git -b muavic clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert && git submodule init && git submodule update
pip install -r requirements.txt
cd fairseq && pip install --editable ./
```
MuAViC's own README is the authority here; follow it if anything differs.

### 2. Data

```bash
conda install -c conda-forge ffmpeg==4.2.2 sox    # their pinned versions
python get_data.py --root-path ${ROOT} --src-lang ru
```
This downloads and aligns 49 h of Russian TED/TEDx talks. Check free disk first.

### 3. MEASURE THE BASELINE FIRST — do not skip this

```bash
bash /path/to/lipflow/scripts/ru/decode_video.sh      # edit the paths at the top first
```

This decodes the **released, untrained-by-you** `ru_avsr` checkpoint on the Russian test set with
`MODALITIES=[video]`, and prints the WER. That number is the thing to beat, and it is the
cheapest possible validation of the whole toolchain — a few hours, no training.

**Decide here.** If the video-only WER comes back around 90–100 %, the checkpoint carries no
usable visual Russian, and fine-tuning starts from nearly nothing: expect a long road and a poor
result. If it comes back at 60–70 %, the visual stream carries real information and the
fine-tuning below is worth the weeks.

### 4. Fine-tune video-only

```bash
bash /path/to/lipflow/scripts/ru/train_video_only.sh   # edit the paths at the top first
```

The one change that matters is `task.modalities=[video]` where MuAViC has `[audio,video]`; the
rest of the diff is single-GPU ergonomics (gradient accumulation, more updates, frequent
checkpoints so a stop is not a loss).

### 5. Re-measure, then export

Run `decode_video.sh` again against your checkpoint. If the WER improved, convert and wire it in:

```bash
python AV-HuBERT-S2S/src/export_muavic_checkpoint_to_hf.py   # fairseq -> transformers
```
The transformers-format model is what `lipflow/asr.py` already loads for GigaAM's sibling, and
an ONNX export of it can be read the same way. Until a checkpoint exists there is nothing to
wire, which is why this repository does not yet ship a Russian visual reader.

## What is deliberately not here

* No training code of our own. MuAViC's recipe is tested; a reimplementation would not be.
* No claim that the result will be good. The measurements above are the honest expectation.

The scripts in this directory were written against MuAViC's `train.sh` and `decode.sh` as
published, but they have **not been run**: developing them needs the GPU machine, not the laptop
this branch was written on. Check the hydra flag names against your checkout of `av_hubert` — the
config file `av_hubert/avhubert/conf/finetune/base_vox_433h.yaml` is the source of truth for
which keys exist.
