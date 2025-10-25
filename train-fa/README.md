## Persian fine-tuning (FA)

### What’s included

* **TSV builder:** `train-fa/build_tsv.py` — creates `data/raw/custom_{train,dev}.tsv` from your CSV + audio folders.
* **Accelerate trainer:** `train-fa/train_zipvoice_fa.py` — HuggingFace Accelerate–based, multi-GPU friendly, supports `--seed` for reproducibility.
* **End-to-end runner:** `train-fa/run_finetune_fa.sh` — pipeline: TSV → manifests → tokens → fbank → download pretrained → fine-tune → average.

### Tokenizer tweak (FA)

* In **`EspeakTokenizer`** (Persian), suprasegmentals are removed **before** tokenization:

  * primary stress `ˈ`, secondary stress `ˌ`
  * vowel length `:` (ASCII; IPA `ː`)
  * eSpeak prosodic boundary marks (phrase/major/minor breaks)
    This reduces token sparsity and stabilizes single-speaker fine-tuning.

### Quick start

```bash
# 1) Clone
%cd /kaggle/working
!rm -rf ZipVoiceFa
!git clone https://github.com/FAIM-TTS/ZipVoiceFa.git
%cd /kaggle/working/ZipVoiceFa

# 2) Dependencies (safe to re-run)
!pip install -q accelerate lhotse cn2an pypinyin piper_phonemize vocos "huggingface_hub[cli]"
```


```bash
%cd /kaggle/working/ZipVoiceFa/egs/zipvoice
!sed -i 's/\r$//' run_finetune_fa.sh

#    #!/bin/bash
!head -n1 run_finetune_fa.sh
!chmod +x run_finetune_fa.sh

!./run_finetune_fa.sh \
  --audio-dirs /kaggle/input/newsinglespk/kaggle/working/m040-ashkan_aghilipour/wavs \
  --csv-path   /kaggle/input/newsinglespk/kaggle/working/m040-ashkan_aghilipour/metadata_all.csv \
  --num-iters 20000 \
  --max-duration 150 \
  --num-epochs 0 \
  --save-every-n 2000 \
  --min-len 0.5 --max-len 25 \
  --gpus 2
```

### Infernece 

```bash
python3 -m zipvoice.bin.infer_zipvoice \
  --model-name zipvoice \
  --model-dir exp/zipvoice_finetune \
  --checkpoint-name best-valid-loss.pt \
  --tokenizer espeak --lang fa \
  --prompt-wav path/to/prompt.wav \
  --prompt-text "متن نمونهٔ پرامپت." \
  --text "سلام بر همه فارسی زبانان جهان." \
  --res-wav-path results_finetune/out.wav \
  --guidance-scale 1 --num-step 16
```
### Result (sample audio)

<audio controls src="https://raw.githubusercontent.com/FAIM-TTS/ZipVoiceFa/master/demo_fa.wav"></audio>
[▶️ Download / open the audio](demo_fa.wav?raw=1)



### Notes

* The runner uses a fixed `--seed` for deterministic runs.
* Use the **`tokens.txt`** produced by this recipe for inference to stay compatible with the tokenizer change.
