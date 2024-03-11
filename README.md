
# ASR Model: HuBERT Encoder + Transformer Decoder [Using SpeechBrain](https://github.com/speechbrain/speechbrain)

This model uses snippets from SpeechBrain's custom TransformerASR (for Decoder) and huggingface_wav2vec (for Encoder) codes.

## HuBERT Encoder:

The model is pretrained (we use the 960h-base version hosted at [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/hubert)), and its weights are stored in **model.bin** file in **/results/save/wav2vec2_checkpoint** folder.

The code for loading and inferencing the HuBERT encoder is stored at **speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2**. It takes in the Raw Waveform and Relative Wavelengths. Finally, it infers the pretrained model using them and returns the encoder output to the training code.

## Transformer Decoder:

The decoder is written at **speechbrain.lobes.models.transformer.HubertASR.HubertASR**. It takes in the encoder output and target. Uses positional encoding, attention mechanism and mask generation, and finally yields decoder output. Decoder output is sent back to the training code for loss calculation.

## Tokenizer & Language Model:

Both are pretrained. We use the same method as SpeechBrain's custom TransformerASR model. Hosted on [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech).

## Training:

The training code is present in **speechbrain.recipes.LibriSpeech.ASR.transformer.train_hubert** and the hyperparameters are present in the hparams subfolder, in **hubert.yaml** file.

Logic is similar to Custom code. Raw Waveforms and Relative Wavelengths are passed to the Encoder (called wav2vec in Module List), which returns the encoder output. This is then passed to the Decoder (called transformer in Module List) and Target (tokens with bos), which returns the Predictions.

These Predictions are used to calculate Seq2Seq log-probabilities. The encoder output is used to calculate CTC log probabilities.

These logits are then passed to the Backward pass for Loss calculation and Backpropagation.

#### Code Flow

![image](https://drive.google.com/uc?export=view&id=1VQd73jrpHXxPmDQNQMykFmGh9gArq_lc)

## Installation

Install the Repository from GitHub

```bash
! git clone https://github.com/midas-research/satyam-btp.git
```

Install the Required Libraries (SpeechBrain and PyTorch)

You need to install the LibriSpeech 960 Hour Dataset. Splits Needed: Train (clean-100, clean-360, other-500), Validation (clean) and Test (clean, other)

```bash
# Train
! wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
! wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
! wget https://www.openslr.org/resources/12/train-other-500.tar.gz

# Dev
! wget https://www.openslr.org/resources/12/dev-clean.tar.gz

# Test
! wget https://www.openslr.org/resources/12/test-clean.tar.gz
! wget https://www.openslr.org/resources/12/test-other.tar.gz
```

The yaml file already contains the default dataset path in data_folder hparam. You'll be able to change it too accordingly.

## Running

To run the code, run the following command

```bash
! cd speechbrain/recipes/LibriSpeech/ASR/transformer/

! python train_hubert.py hparams/hubert.yaml
```

---
\
**Note:** Inside the YAML file, the Gradient Accumulation factor is 4, and the Batch Size is 32. This will be enough to simulate the performance of 4 GPUs on a single 40GB GPU. You can change them according to the number of GPUs you use.

Ensure that => **Batch Size X Grad Factor X Num GPUs >= 128**.
