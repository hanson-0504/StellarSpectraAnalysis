# agents/llm_agent.py
import os, json, time, yaml, logging
from typing import Any, Dict
from datetime import datetime

from .state import load_state, save_state
from . import tools

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "preprocess_spectra",
            "description": "Preprocess raw FITS spectra and labels into flux arrays.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fits_dir": {"type": "string"},
                    "labels_dir": {"type": "string"}
                },
                "required": []
            }
        },
    },
    {
        "type": "function",
        "function": {"name": "train_models", "description": "Train ML models.", "parameters": {"type": "object", "properties": {}}},
    },
    {
        "type": "function",
        "function": {
            "name": "predict_new",
            "description": "Run predictions on spectra using trained models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fits_dir": {"type": "string"},
                    "out_dir": {"type": "string"}
                },
                "required": []
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "evaluate_results", "description": "Compute metrics and residual summaries.", "parameters": {"type": "object", "properties": {}}},
    },
]

def llm_loop(poll_seconds: int = 10, max_steps: int = 10):
    client = OpenAI()  # needs OPENAI_API_KEY in env
    with open("config.yaml", "r") as f:
        CFG = yaml.safe_load(f)
    state = load_state()

    sys_prompt = (
        "You are an ML pipeline orchestrator for stellar spectra. "
        "Choose one tool per step. Only choose a tool if it is needed; otherwise say 'idle'. "
        "Prefer preprocess → train → predict → evaluate, reacting to what changed."
    )

    for step in range(max_steps):
        # Summarize current world state for the model (keep it short)
        world = {
            "time": datetime.utcnow().isoformat(),
            "state": {k: state.get(k) for k in ["last_preprocess", "last_train", "last_predict"]},
            "dirs": CFG["directories"],
        }

        msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Current state: {json.dumps(world)}\nWhat should we do next?"}
        ]

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=msg,
            tools=TOOL_SPECS,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = resp.choices[0].message

        if choice.tool_calls:
            call = choice.tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")

            logging.info(f"[llm] chose tool: {name}({args})")

            # Dispatch to your actual tools
            result: Dict[str, Any]
            if name == "preprocess_spectra":
                result = tools.preprocess_spectra(**args)
                state["last_preprocess"] = time.time()
            elif name == "train_models":
                result = tools.train_models()
                state["last_train"] = time.time()
            elif name == "predict_new":
                result = tools.predict_new(**args)
                state["last_predict"] = time.time()
            elif name == "evaluate_results":
                result = tools.evaluate_results()
            else:
                result = {"status": "unknown_tool"}

            save_state(state)

            # Return result to the model (optional follow-up)
            follow = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msg + [
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                ],
                temperature=0.2,
            )
        else:
            # Model says to idle
            logging.info("[llm] idle")
            time.sleep(poll_seconds)
