import os
import subprocess
import sys
from pathlib import Path


def test_importing_logger_does_not_create_files(local_tmpdir: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    subprocess.run(
        [sys.executable, "-c", "import src.core.logger"],
        cwd=local_tmpdir,
        env=environment,
        check=True,
    )

    assert not (local_tmpdir / "logs").exists()


def test_first_log_message_configures_a_file(local_tmpdir: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.core.logger import log_info; log_info('configured lazily')",
        ],
        cwd=local_tmpdir,
        env=environment,
        check=True,
    )

    log_files = list((local_tmpdir / "logs").glob("analysis_*.log"))
    assert len(log_files) == 1
    assert "configured lazily" in log_files[0].read_text(encoding="utf-8")
