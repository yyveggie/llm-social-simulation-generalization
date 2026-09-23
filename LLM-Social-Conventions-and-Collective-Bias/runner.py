from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
AUTHOR_CODE = ROOT / "author_code"
if str(AUTHOR_CODE) not in sys.path:
    sys.path.insert(0, str(AUTHOR_CODE))

runpy.run_path(str(AUTHOR_CODE / "runner.py"), run_name="__main__")
