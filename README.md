# Maral NMT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcst-thesis/maral-onnx-nmt/blob/main/notebooks/maral_nmt_colab.ipynb)

Maral NMT trains a bidirectional sequence-to-sequence translation model on one NVIDIA GPU.
It accepts a parallel corpus with two columns. It exports TorchScript and a
self-contained, dynamically quantized INT8 ONNX model for greedy mobile
decoding.

## Run the Notebook in Google Colab

1. Open the notebook with the badge.
2. Select **Runtime → Change runtime type → GPU**.
3. Add a tab-separated corpus to Google Drive.
4. Put the two column names in the first row.

   ```tsv
   source	target
   Hello.	Kumusta.
   ```

5. Set the file path and column names in the configuration cell.
6. Set the language names and output directory.
7. Run all cells in sequence.

The notebook saves the best checkpoint and the tokenizer. It also saves the
training history, configuration, TorchScript model, and `nmt_model.onnx`.
The ONNX model returns one `next_token` ID per call instead of copying the full
vocabulary logits to the mobile client.

### Mobile ONNX Contract

- `src_tokens`: `int64[batch, source_length]`
- `tgt_tokens`: `int64[batch, target_length]`
- `next_token`: `int64[batch]`

Encode source text with `tokenizer.model`, prefixing it with `<2target>` for
source-to-target translation or `<2source>` for target-to-source translation.
Start the output with BOS ID `2`, append each `next_token`, and stop at EOS ID
`3` or the configured decode limit. Both directions reuse the same tokenizer,
ONNX model, and ONNX Runtime session.

## Validate the Notebook

Run this command from the repository root:

```bash
python scripts/validate_notebook.py
```

Training requires a CUDA-capable GPU runtime. The notebook uses the
CUDA-enabled PyTorch installation from Colab.
