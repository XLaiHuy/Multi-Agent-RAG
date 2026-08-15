"""
Compilation Regression Test Suite.
Verifies that all Python sources under backend/, evaluation/, tests/, and scripts/ compile cleanly.
"""
import compileall
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_all_python_sources_compile_cleanly():
    """Ensures zero syntax errors or compilation failures across the entire codebase."""
    target_dirs = ["backend", "evaluation", "tests", "scripts"]
    for dir_name in target_dirs:
        target_path = REPO_ROOT / dir_name
        if not target_path.exists():
            continue
        # compile_dir returns True if all files compiled successfully, False otherwise
        success = compileall.compile_dir(
            str(target_path),
            quiet=1,
            force=True,
            legacy=False,
            optimize=0
        )
        assert success is True, f"Compilation failed for directory: {dir_name}"
