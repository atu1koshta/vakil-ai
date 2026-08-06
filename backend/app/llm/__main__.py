"""Smoke test + parametric-knowledge baseline: what does the model claim
WITHOUT retrieval? Keep an example handy — the "before" picture for the
grounding lesson.

    python -m app.llm "What did the Supreme Court hold in Chintaman Rao?"
    python -m app.llm --model deepseek "same question..."
"""

import sys

from . import get_chat_model

args = sys.argv[1:]
model_name = None
if args[:1] == ["--model"] and len(args) >= 2:
    model_name, args = args[1], args[2:]

question = " ".join(args) or "In one sentence, what is Article 19(1)(g)?"
print(get_chat_model(model_name).chat("You are a legal assistant. Be concise.", question))
