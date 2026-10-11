"""Run only on a local developer machine: python run_server.py"""
from pathlib import Path
import uvicorn
from orgsim.api import create_app
app = create_app(Path(__file__).parent / "manifest.json", Path(__file__).parent / "runtime.sqlite")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
