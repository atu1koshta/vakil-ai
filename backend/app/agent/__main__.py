"""Agent smoke test from the terminal — watch the loop think in tools.

    python -m app.agent --model llama "Which judgments are from the Bombay High Court?"
    python -m app.agent --model llama --max-steps 4 "What did Chintaman Rao strike down?"

The active model (deepseek-r1) has no tool support in Ollama — expect the
actionable GenerationError unless --model llama / deepseek-api is passed.
"""

import json
import sys

from . import get_agent

args = sys.argv[1:]
model_name = None
max_steps = 6
while args[:1] and args[0].startswith("--"):
    if args[0] == "--model" and len(args) >= 2:
        model_name, args = args[1], args[2:]
    elif args[0] == "--max-steps" and len(args) >= 2:
        max_steps, args = int(args[1]), args[2:]
    else:
        sys.exit(f"unknown flag {args[0]}")

question = " ".join(args) or "Which judgments discuss reasonable restrictions on trade?"
result = get_agent().run(question, model=model_name, max_steps=max_steps)

for i, step in enumerate(result.steps, 1):
    flag = " [ERROR]" if step.error else ""
    print(f"#{i} {step.tool}({json.dumps(step.args)}) — {step.duration_ms}ms{flag}")
    print(f"   {step.result_preview[:160]}".replace("\n", " "))
print()
if result.exhausted:
    print("[exhausted — forced answer]")
print(result.answer)
