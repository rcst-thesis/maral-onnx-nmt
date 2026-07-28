"""Validate the committed Colab notebook using only the Python standard library."""

import ast
import json
from pathlib import Path


notebook_path = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "maral_nmt_colab.ipynb"
)
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

assert notebook["nbformat"] == 4
assert notebook["metadata"]["accelerator"] == "GPU"
assert notebook["metadata"]["colab"]["include_colab_link"] is True

notebook_source = "\n".join(
    "".join(cell["source"])
    for cell in notebook["cells"]
)
for required_text in (
    "torch.cuda.is_available()",
    'torch.amp.autocast("cuda"',
    'torch.amp.GradScaler("cuda")',
    "pin_memory=True",
    "non_blocking=True",
):
    assert required_text in notebook_source

for forbidden_text in (
    "torch_xla",
    "MpDeviceLoader",
    "run_xla_epoch",
):
    assert forbidden_text not in notebook_source

for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    assert cell["execution_count"] is None
    assert cell["outputs"] == []
    source = "".join(cell["source"])
    python_source = "\n".join(
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    try:
        ast.parse(python_source)
    except SyntaxError as error:
        raise SyntaxError(f"Notebook cell {index}: {error}") from error

print(f"Validated {len(notebook['cells'])} notebook cells.")
