"""Package-wide rules: the core imports no model library, names nothing of Anvil,
keeps a README in every folder, its version matches pyproject.toml, and README.md's examples run."""

import os
import pathlib
import re
import subprocess
import sys
import tomllib

import contextattr

SRC = pathlib.Path(contextattr.__file__).parent
MODEL_LIBRARIES = sorted({"torch", "transformers", "vllm", "sglang", "openai", "anthropic", "boto3", "botocore",
                          "google", "tensorflow", "jax", "requests", "httpx"})
# Imported only when a method needs them: their imports take 18 s (scikit-learn) to 25–45 s (shapiq) on a login node.
LAZY = sorted({"sklearn", "scipy", "shapiq", "galois", "numba", "lightgbm"})
# Account and allocation names are not spelled out here, as this file is published: the release check reads them
# from a list that is not published and checks every built file for them.
ANVIL_WORDS = ("/anvil/", "slurm", "sbatch", "module load", "rcac")


def test_importing_the_package_loads_no_model_library_and_nothing_slow():
    code = (f"import sys, contextattr; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({MODEL_LIBRARIES + LAZY!r})))")
    env = {**os.environ, "PYTHONPATH": str(SRC.parent)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip() == "[]"


def test_the_package_names_nothing_of_anvil():
    hits = [f"{p.relative_to(SRC)}: {word}" for p in SRC.rglob("*.py")
            for word in ANVIL_WORDS if word in p.read_text(encoding="utf-8").lower()]
    assert hits == []


def test_every_package_folder_has_a_readme():
    folders = [SRC, *(d for d in SRC.rglob("*") if d.is_dir() and d.name != "__pycache__")]
    assert [str(d.relative_to(SRC.parent)) for d in folders if not (d / "README.md").is_file()] == []


def test_version_matches_pyproject():
    with open(SRC.parent.parent / "pyproject.toml", "rb") as f:
        assert tomllib.load(f)["project"]["version"] == contextattr.__version__


def test_the_readme_runs_and_names_nothing_of_anvil(capsys):
    """README.md is the package's page on PyPI: its example runs and gives what the text says it gives; it names
    nothing of Anvil."""
    readme = (SRC.parent.parent / "README.md").read_text(encoding="utf-8")
    assert [w for w in ANVIL_WORDS if w in readme.lower()] == []
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert len(blocks) == 1
    namespace: dict = {}
    exec(compile(blocks[0], "README.md", "exec"), namespace)  # noqa: S102 - the README's own example
    first, *rows = capsys.readouterr().out.strip().splitlines()
    e = namespace["e"]                                       # "d2 scores highest"; "the first line gives R²"
    assert e.top(1)[0][0] == "d2" and rows[1].split()[0] == "d2" and "R² " in first and e.r2 > 0.95
    assert rows[0].strip().startswith("above 0: the unit makes the answer more likely")
