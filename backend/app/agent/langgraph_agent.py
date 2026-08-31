"""The 3a loop as a LangGraph graph — a translation, plus what came free.

Learning notes:
- Node-for-line mapping to hand_rolled.py: the `agent` node = the
  chat_tools() call; ToolNode = the for-tc-in-tool_calls block (execution,
  and with handle_tool_errors left OFF, our ERROR-string semantics: user
  mistakes come back as tool messages via run_tool, infra errors
  propagate); tools_condition = `if not result.tool_calls`; the
  recursion_limit + GraphRecursionError handler = max_steps + the forced
  final answer.
- What the framework replaces outright: sessions.py. MemorySaver keyed by
  thread_id (= our session_id) checkpoints the WHOLE message state after
  every superstep — including tool messages, which the hand-rolled store
  deliberately trims. Framework memory is completer and dumber; the trim
  problem returns as a context-window problem at scale, solved there by
  trimming/summarization nodes. AgentResult.history stays empty here so
  the endpoint knows not to double-store.
- The graph is rebuilt per run (cheap — a dict of nodes) because tools are
  closures over this run's profile + trace (lg_tools.py). The CHECKPOINTER
  is the module-level singleton: state lives in it, not in the compiled
  graph, so chat threads survive rebuilds — exactly the registry/index
  split again (durable state vs disposable wiring).
- The LangChain model wrappers (ChatOllama/ChatOpenAI) replace our
  ChatModel providers INSIDE this agent only: they do the same
  _to_wire()/tool_calls translation llm/ does by hand. Both stacks stay —
  comparing them is the point of 3b.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import ConfigError, GenerationConfig, get_chat_config
from ..llm import GenerationError
from .base import Agent, AgentError, AgentResult, AgentStep
from .hand_rolled import AGENT_SYSTEM_PROMPT, FORCED_ANSWER_PROMPT
from .lg_tools import build_tools
from .sessions import new_session_id

# One checkpointer for the process: chat threads (thread_id = session_id)
# live here across requests, like sessions._sessions for the raw loop.
_CHECKPOINTER = MemorySaver()


def _make_llm(cfg: GenerationConfig):
    """GenerationConfig -> LangChain chat model. Mirrors llm/ providers."""
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            num_ctx=cfg.num_ctx,
        )
    if cfg.provider == "openai-compatible":
        from langchain_openai import ChatOpenAI

        if not cfg.api_key_env:
            raise GenerationError(
                f"chat model '{cfg.name}' uses an API provider but sets "
                "no api_key_env in config.yaml"
            )
        key = os.environ.get(cfg.api_key_env, "")
        if not key:
            raise GenerationError(
                f"env var {cfg.api_key_env} is not set — "
                f"export {cfg.api_key_env}=<your key> before starting"
            )
        return ChatOpenAI(
            model=cfg.model,
            base_url=f"{cfg.base_url}/v1",
            api_key=key,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
        )
    raise ConfigError(
        f"no LangChain wrapper for provider '{cfg.provider}' "
        "(known: ollama, openai-compatible)"
    )


def _build_graph(llm, tools: list[BaseTool]):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        # System prompt is prepended per call, never checkpointed — stored
        # once per thread it would duplicate on every turn.
        response = llm_with_tools.invoke(
            [SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool_calls ? tools : END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=_CHECKPOINTER)


class LangGraphAgent(Agent):
    def run(
        self,
        question: str,
        *,
        model: str | None = None,
        profile: str | None = None,
        max_steps: int = 6,
        history: list[dict] | None = None,  # unused: the checkpointer owns memory
        session_id: str | None = None,
    ) -> AgentResult:
        cfg = get_chat_config(model)
        trace: list[AgentStep] = []
        graph = _build_graph(_make_llm(cfg), build_tools(profile, trace))
        config = {
            "configurable": {"thread_id": session_id or new_session_id()},
            # agent+tools per tool round, +1 for the closing agent call.
            "recursion_limit": 2 * max(1, max_steps) + 1,
        }

        prior_state = graph.get_state(config)
        prior_count = len((prior_state.values or {}).get("messages", []))

        exhausted = False
        try:
            state = graph.invoke({"messages": [HumanMessage(question)]}, config)
            messages = state["messages"]
        except GraphRecursionError:
            # Same recovery as the raw loop: one call with tools DISABLED
            # forces an answer from gathered evidence. The checkpointer has
            # every superstep up to the limit; write the answer back so the
            # thread stays coherent for the next turn.
            exhausted = True
            messages = graph.get_state(config).values["messages"]
            forced = _make_llm(cfg).invoke(
                [
                    SystemMessage(AGENT_SYSTEM_PROMPT),
                    *messages,
                    HumanMessage(FORCED_ANSWER_PROMPT),
                ]
            )
            graph.update_state(config, {"messages": [forced]})
            messages = [*messages, forced]

        answer = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage) and m.content
            ),
            "",
        )
        if not answer:
            raise AgentError("model produced no final answer")
        return AgentResult(
            question=question,
            answer=answer if isinstance(answer, str) else str(answer),
            model=cfg.name,
            steps=trace,
            iterations=sum(
                isinstance(m, AIMessage) for m in messages[prior_count:]
            ),
            exhausted=exhausted,
            history=[],  # memory lives in the checkpointer, keyed by session_id
        )
