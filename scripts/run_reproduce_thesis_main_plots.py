"""Execute the thesis main-plots notebook from the command line.

This keeps the notebook as the single source of truth: the setup cell stages
the data, and the three plotting cells generate both the full and split figures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "reproduce_thesis_main_plots.ipynb"
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    os.chdir(repo_root)
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}

    for idx, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        print(f"\n=== Running notebook cell {idx} ===")
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path.name}:cell{idx}", "exec"), namespace)

    print("\nDone. Full and split thesis plots have been generated.")


if __name__ == "__main__":
    main()
