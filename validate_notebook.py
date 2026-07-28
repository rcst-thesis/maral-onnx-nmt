"""Validate the committed Colab notebook using only the Python standard library."""

import ast
import json
from pathlib import Path


notebook = json.loads(Path("maral_nmt_colab.ipynb").read_text(encoding="utf-8"))

assert notebook["nbformat"] == 4
assert notebook["metadata"]["accelerator"] == "TPU"
assert notebook["metadata"]["colab"]["include_colab_link"] is True

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
