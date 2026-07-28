# Maral NMT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcst-thesis/maral-onnx-nmt/blob/main/maral_nmt_colab.ipynb)

A general-purpose, single-TPU neural machine translation trainer for
two-column parallel corpora. It trains a shared SentencePiece tokenizer and an
AttnRes sequence-to-sequence model, then exports TorchScript and ONNX models.

## Run in Google Colab

1. Open the notebook with the badge above.
2. Select **Runtime → Change runtime type → TPU**.
3. Put a tab-separated corpus in Google Drive. The first row must contain two
   column names, for example:

   ```tsv
   source	target
   Hello.	Kumusta.
   ```

4. Set the dataset path, column names, language labels, and output directory in
   the notebook's configuration cell.
5. Run the notebook from top to bottom.

The best checkpoint, tokenizer, training history, TorchScript model, ONNX
model, and configuration are copied to the configured Drive output directory.

## Local validation

```bash
python validate_notebook.py
```

Training requires a PyTorch/XLA TPU runtime. The notebook intentionally relies
on Colab's matching PyTorch and PyTorch/XLA installation.
