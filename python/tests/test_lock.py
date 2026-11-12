"""The Python versions the project allows against the wheels it locks.

The Host's uv makes the environment with the newest managed Python `requires-python` allows, so every Python it
allows must find a wheel of every locked package built for one Python only (`cp312`, …): ONNX Runtime's builds,
numpy and Pillow. Python 3.15 came out before ONNX Runtime had wheels for it, and an uncapped `requires-python` made
every build uninstallable on a fresh Host.
"""

import re
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LOCK = tomllib.loads((PROJECT / "uv.lock").read_text(encoding="utf-8"))
PYPROJECT = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))


def allowed_minors(spec: str) -> list[int]:
    """The Python 3 minors a `>=3.A, <3.B` range allows. Fails on a range without a cap: the newest Python is then
    whatever was released last."""
    low = re.search(r">=\s*3\.(\d+)", spec)
    high = re.search(r"<\s*3\.(\d+)", spec)
    assert low and high, f"requires-python {spec!r} needs a floor and a cap: >=3.A, <3.B"
    return list(range(int(low.group(1)), int(high.group(1))))


def cpython_tags(package: dict) -> set[str] | None:
    """The Pythons a package's wheels are built for (`cp314`, free-threaded ones left out), or None when a wheel serves
    any Python (`py3`, `abi3`) or the package has no wheels."""
    tags = set()
    for wheel in package.get("wheels", []):
        name = wheel["url"].rsplit("/", 1)[-1]
        _, _, python, abi, _ = name[: -len(".whl")].rsplit("-", 4)
        if not python.startswith("cp") or abi in ("abi3", "none"):
            return None
        if not abi.endswith("t"):
            tags.add(python)
    return tags or None


def test_requires_python_is_the_same_in_the_project_and_its_lock():
    assert PYPROJECT["project"]["requires-python"].replace(" ", "") == LOCK["requires-python"].replace(" ", "")


def test_every_python_allowed_has_a_wheel_of_every_package_built_per_python():
    minors = allowed_minors(LOCK["requires-python"])
    missing = {}
    for package in LOCK["package"]:
        tags = cpython_tags(package)
        if tags is None:
            continue
        lacking = [f"cp3{minor}" for minor in minors if f"cp3{minor}" not in tags]
        if lacking:
            missing[f"{package['name']} {package['version']}"] = lacking
    assert missing == {}, f"no wheels for these allowed Pythons: lower requires-python's cap: {missing}"


def test_the_onnx_runtime_builds_are_among_the_packages_checked():
    checked = {package["name"] for package in LOCK["package"] if cpython_tags(package) is not None}
    assert {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "numpy"} <= checked
