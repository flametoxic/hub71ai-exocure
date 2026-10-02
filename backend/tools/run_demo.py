"""Start the demo on Windows, macOS or Linux: python tools/run_demo.py."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from arrival.config import load_environment
    load_environment()
    # Child processes use the same interpreter and can import the vendored core.
    os.environ["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    os.environ.setdefault("PYTHONUTF8", "1")
    defaults = {
        "CURE_LANGUAGE_MODE": "auto",
        "ARRIVAL_TEMPLATE_KEY": "demo-template-key-change-me",
        "ARRIVAL_VAULT_KEY": "demo-vault-key-change-me-0001",
        "ARRIVAL_CORE_KEY": "demo-core-key-change-me-00001",
        "ARRIVAL_CORE_DIR": str(ROOT / "core_store"),
    }
    for name, value in defaults.items():
        if not os.environ.get(name):
            os.environ[name] = value

    missing = [name for name in ("fastapi", "uvicorn", "numpy", "yaml", "cryptography", "pydantic")
               if importlib.util.find_spec(name) is None]
    if (os.environ.get("CURE_LANGUAGE_MODE") == "openai" or
            (os.environ.get("CURE_LANGUAGE_MODE") == "auto" and os.environ.get("OPENAI_API_KEY"))):
        if importlib.util.find_spec("openai") is None:
            missing.append("openai")
    if missing:
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        print('Install with: python -m pip install -r requirements.txt', file=sys.stderr)
        return 1
    port_text = os.environ.get("PORT") or "8000"
    try:
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        print("PORT must be an integer between 1 and 65535.", file=sys.stderr)
        return 1

    try:
        if not (ROOT / "arrival/demo/policy.yaml").is_file():
            subprocess.run([sys.executable, "tools/gen_demo_data.py"], check=True)
        subprocess.run([sys.executable, "-m", "arrival.sign_templates", "arrival/demo/agents.yaml"], check=True)
        print(f"CURE API: http://localhost:{port}/docs", flush=True)
        return subprocess.call([
            sys.executable, "-m", "uvicorn", "demo_app:app",
            "--host", os.environ.get("HOST") or "0.0.0.0", "--port", str(port),
        ])
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
