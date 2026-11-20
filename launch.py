"""Install the local CPU tool on first launch and start its modeling UI."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--torch-wheel", type=Path, help="Use an already downloaded CPU PyTorch wheel")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.exit(1, "Install Python 3.10 or newer (64-bit).\n")

    root = Path(__file__).resolve().parent
    home = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    state = home / "DSpaLaRefiner"
    env = state / "cpu-0.1.1" / sys.implementation.cache_tag
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = env / ".ready"
    setup_env = dict(os.environ, PIP_NO_CACHE_DIR="1", PIP_TIMEOUT="25", PIP_RETRIES="2")
    try:
        if not marker.is_file() or not executable.is_file():
            print("Setting up the local CPU environment. First launch needs internet.", flush=True)
            venv.EnvBuilder(with_pip=True).create(env)
            pip = [str(executable), "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "--timeout", "25", "--retries", "2"]
            subprocess.run([*pip, "install", "--upgrade", "pip>=25"], check=True, env=setup_env)
            torch = [str(args.torch_wheel.resolve())] if args.torch_wheel else [
                "torch>=2.0,<3", "--index-url", "https://download.pytorch.org/whl/cpu",
            ]
            print("Installing CPU PyTorch. This is the largest download.", flush=True)
            subprocess.run([*pip, "install", *torch], check=True, env=setup_env)
            subprocess.run([*pip, "install", "setuptools>=68", "wheel"], check=True, env=setup_env)
            print("Installing DSpaLaRefiner and its other dependencies...", flush=True)
            # Build from a temporary copy so imports and Git stay clean.
            with tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "package"
                package.mkdir()
                for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"):
                    shutil.copy2(root / name, package / name)
                shutil.copytree(root / "tabular_agent", package / "tabular_agent",
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                subprocess.run([*pip, "install", "--no-build-isolation", str(package)], check=True, env=setup_env)
            marker.touch()
        if args.prepare_only:
            return 0
        command = [
            str(executable), "-m", "tabular_agent.serve",
            "--port", str(args.port), "--runs", str(state / "runs"),
        ]
        if not args.no_browser:
            command.append("--open-browser")
        return subprocess.call(command, cwd=state)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Setup failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())

