# agents/state.py
import json
from pathlib import Path
from typing import Any, Dict

STATE_FILE = Path("logs/agent_state.json")

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_preprocess": 0.0, "last_train": 0.0, "last_predict": 0.0, "runs": []}

def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
