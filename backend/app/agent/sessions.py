"""In-memory chat sessions — the hand-rolled version of a checkpointer.

Learning notes:
- What persists between turns is DELIBERATELY not the full transcript:
  only user questions + final answers. Tool results are bulky (400+ chars
  each, many per turn) and go stale — the agent can always re-fetch. This
  is the message-trimming problem every framework has a node for; felt
  raw here so 3b's checkpointer + trim reads as a translation.
- MAX_TURNS caps history so a long chat can't crowd the model's context
  window out of tool results for the CURRENT question.
- A plain dict behind a lock = studio-grade. Restart wipes every chat;
  that pain is exactly what a durable checkpointer (sqlite saver) removes
  in 3b.
"""

import threading
import uuid

MAX_TURNS = 10  # one turn = user question + final assistant answer

_lock = threading.Lock()
_sessions: dict[str, list[dict]] = {}


def new_session_id() -> str:
    return uuid.uuid4().hex


def get_history(session_id: str) -> list[dict]:
    """Prior turns for a session; [] for an unknown id (new or post-restart)."""
    with _lock:
        return list(_sessions.get(session_id, []))


def save_history(session_id: str, history: list[dict]) -> None:
    """Persist a session's turns, keeping only the most recent MAX_TURNS."""
    with _lock:
        _sessions[session_id] = history[-2 * MAX_TURNS :]
