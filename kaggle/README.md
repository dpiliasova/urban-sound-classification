# Kaggle GPU run

`dev_run.py` performs the first fresh experiment after the preprocessing audit:

- clones the current public project;
- locates the attached UrbanSound8K dataset without relying on one fixed mount
  path;
- preloads one shared copy of the corrected spectrogram features;
- trains on folds 1–8, selects the checkpoint on fold 9, and evaluates fold 10;
- records the Git commit, package versions, GPU, runtime, metrics, predictions,
  history, confusion matrix, and best checkpoint;
- removes the temporary feature cache only after a successful run.

The Kaggle kernel is private until the corrected run has completed and its
outputs have been reviewed.

The `cv/` directory contains a separate private kernel for the corrected
ten-fold experiment. It reuses one in-memory spectrogram cache across folds and
removes fold checkpoints after evaluation while retaining metrics, histories,
predictions, confusion matrices, and the aggregate summary.

Upload a new version with the Kaggle CLI:

```bash
kaggle kernels push -p kaggle
```

Check its state and download outputs:

```bash
kaggle kernels status dashapilyasova/urbansound8k-corrected-densenet-dev-run
kaggle kernels output dashapilyasova/urbansound8k-corrected-densenet-dev-run -p artifacts/kaggle-dev
```
