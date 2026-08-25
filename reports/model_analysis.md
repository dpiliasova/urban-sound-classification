# Model analysis

## Problem and information boundary

UrbanSound8K contains 8,732 audio excerpts from ten urban-sound classes. The
goal is multiclass classification from at most four seconds of audio. Each
prediction uses only the target clip; no filename, fold identifier, source ID,
or other metadata enters the model.

The dataset authors provide ten folds that keep excerpts from the same source
recording together. The project preserves those folds because randomly moving
clips between splits could place related excerpts in training and testing and
inflate quality.

## Validation protocol

Architecture and training decisions were developed on a fixed split:

- train: folds 1–8;
- validation: fold 9;
- test: fold 10.

For the recorded final experiment, each official fold was used once as test.
The next fold cyclically was used for checkpoint selection, and the remaining
eight folds were used for training. Test folds were not used for early stopping.

| Test fold | Validation fold | Accuracy |
|---:|---:|---:|
| 1 | 2 | 78.58% |
| 2 | 3 | 85.14% |
| 3 | 4 | 72.22% |
| 4 | 5 | 86.26% |
| 5 | 6 | 82.26% |
| 6 | 7 | 80.19% |
| 7 | 8 | 81.74% |
| 8 | 9 | 77.17% |
| 9 | 10 | 84.56% |
| 10 | 1 | 82.92% |
| **Mean ± population SD** | — | **81.10% ± 4.02%** |

The 14-point range between folds 3 and 4 is large enough that reporting only a
favourable development split would be misleading.

## Architecture and training

The recorded experiment used:

- 128-bin log-mel spectrograms at 22,050 Hz;
- ImageNet-pretrained DenseNet121 with a ten-class head;
- head-only warm-up followed by full-backbone fine-tuning;
- AdamW, label smoothing, Mixup, time/frequency masks, time shifts, and noise;
- validation-based early stopping and best-checkpoint restoration.

Transfer learning and two-stage fine-tuning gave the largest observed gain over
the custom CNN attempted during coursework. The individual contribution of
every augmentation was not measured in a controlled ablation, so causal claims
about augmentation gains are intentionally avoided.

## Fixed-split error analysis

The development test fold reached 84.71% accuracy and 84.93% weighted F1.

- `gun_shot` was the strongest class: 96.88% precision and recall, but only 32
  test examples were available.
- `children_playing` reached 92.00% recall but 65.25% precision, indicating
  substantial overprediction.
- `jackhammer` showed a similar pattern: 93.75% recall and 74.38% precision.
- `drilling` and `dog_bark` had lower recall, at 73.00% and 75.00%.
- The original confusion-matrix review suggested plausible confusion among
  sustained mechanical sounds (`drilling`, `jackhammer`, `engine_idling`, and
  `air_conditioner`). Saved per-example predictions in the refactored runner
  make this claim directly reproducible on the next run.

## Portfolio audit and correction

The coursework dataset class set SpecAugment masks to zero on decibel-valued
spectrograms and normalized afterward. Since the maximum decibel value is zero,
masked regions could become high-valued rather than neutral. The refactored
pipeline now:

1. computes the log-mel spectrogram;
2. standardizes it per example;
3. applies masks with the normalized mean, zero.

The old 81.10% result therefore documents the original experiment, not a claim
that the corrected package has already reproduced it. A fresh ten-fold GPU run
is the next required experiment.

## Limitations and next experiments

- rerun the corrected pipeline and compare it with the historical result;
- repeat at least the development split across several seeds;
- run a controlled augmentation ablation;
- compare repeated-channel DenseNet with an audio-pretrained backbone;
- inspect per-class metrics across all folds, not only fold 10;
- evaluate calibration and confidence on out-of-distribution recordings.
