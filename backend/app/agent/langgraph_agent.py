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
- Corrective RAG (the grade node): after a search_chunks round, a second
  LLM call grades whether the passages answer the question; insufficient
  verdicts append a grader message (what's missing + ONE rewritten query)
  and loop back to the agent. This is the gated version of the 2c lesson:
  blind per-query rewrite measured net-negative (false-consensus tax), so
  rewrite only fires when grading says retrieval FAILED — and at most
  MAX_REWRITES times, because a grader that keeps saying "not enough"
  must not be able to spend the whole step budget. The grader fails OPEN
  (treated as sufficient): grading is an optimization, never the reason
  an answer didn't come back.
"""

import os
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from ..config import ConfigError, GenerationConfig, get_chat_config
from ..llm import GenerationError
from .base import Agent, AgentError, AgentResult, AgentStep
from .hand_rolled import AGENT_SYSTEM_PROMPT, FORCED_ANSWER_PROMPT
from .lg_tools import PREVIEW_CHARS, build_tools
from .sessions import new_session_id

# One checkpointer for the process: chat threads (thread_id = session_id)
# live here across requests, like sessions._sessions for the raw loop.
_CHECKPOINTER = MemorySaver()

MAX_REWRITES = 2  # grader-triggered retries per turn; after this, answer from what's there
EVIDENCE_CAP = 6000  # chars of search results shown to the grader


class AgentState(MessagesState):
    """MessagesState + this turn's rewrite budget. `rewrites` uses the
    default overwrite reducer and is reset to 0 by every invoke input, so
    the budget is per TURN even though the checkpointer persists it."""

    rewrites: int


class Grade(BaseModel):
    """Grader verdict — structured output, so routing never parses prose."""

    sufficient: bool = Field(
        description="true when the passages contain enough evidence to answer"
    )
    missing: str = Field(
        default="", description="when insufficient: what evidence is missing"
    )
    rewritten_query: str = Field(
        default="",
        description=(
            "when insufficient: ONE alternative search query using different "
            "words — synonyms, legal terms, section numbers"
        ),
    )


GRADE_PROMPT = """You are grading retrieval results for a legal research question.

Question: {question}

Retrieved passages:
{evidence}

Decide whether these passages contain enough evidence to answer the question.
Mark sufficient=true when they answer it, even partially but usefully.
Mark sufficient=false ONLY when they are off-topic or miss the core of the
question — then state what is missing and give ONE rewritten search query
using different words (synonyms, legal terms, section numbers)."""

GRADER_GUIDANCE = (
    "[retrieval grader] The search results so far do not answer the "
    "question: {missing} Search again with a DIFFERENT query — try "
    "{query!r} — or use another tool if better. Do not repeat a previous "
    "query."
)


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


def _build_graph(llm, tools: list[BaseTool], trace: list[AgentStep], question: str):
    llm_with_tools = llm.bind_tools(tools)
    # function_calling, not the default json_schema response_format: the
    # agent's model must support tool calling anyway, and deepseek-chat
    # rejects json_schema ("This response_format type is unavailable").
    grader = llm.with_structured_output(Grade, method="function_calling")

    def agent_node(state: AgentState) -> dict:
        # System prompt is prepended per call, never checkpointed — stored
        # once per thread it would duplicate on every turn.
        response = llm_with_tools.invoke(
            [SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    def route_after_tools(state: AgentState) -> str:
        """Grade only search rounds, and only while rewrite budget remains —
        filter/read results have nothing to grade, and a spent budget skips
        the grader LLM call entirely instead of grading into the void."""
        if state.get("rewrites", 0) >= MAX_REWRITES:
            return "agent"
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                if any(tc["name"] == "search_chunks" for tc in m.tool_calls):
                    return "grade"
                return "agent"
        return "agent"

    def grade_node(state: AgentState) -> dict:
        # Evidence = this round's search results: the trailing run of
        # ToolMessages (everything after the last AIMessage).
        evidence: list[str] = []
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name == "search_chunks":
                evidence.append(str(m.content))
        evidence_text = "\n\n".join(reversed(evidence))[:EVIDENCE_CAP]

        rewrites = state.get("rewrites", 0)
        started = time.perf_counter()
        failed = False
        try:
            verdict = grader.invoke(
                GRADE_PROMPT.format(question=question, evidence=evidence_text)
            )
        except Exception as exc:  # fail open — grading must never block an answer
            verdict = Grade(sufficient=True, missing=f"grader failed: {exc}")
            failed = True
        trace.append(
            AgentStep(
                tool="grade_retrieval",
                args={"rewrites_used": rewrites},
                result_preview=verdict.model_dump_json()[:PREVIEW_CHARS],
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=failed,
            )
        )
        if verdict.sufficient:
            return {}
        guidance = HumanMessage(
            GRADER_GUIDANCE.format(
                missing=verdict.missing, query=verdict.rewritten_query
            )
        )
        return {"messages": [guidance], "rewrites": rewrites + 1}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_node("grade", grade_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool_calls ? tools : END
    graph.add_conditional_edges(
        "tools", route_after_tools, {"grade": "grade", "agent": "agent"}
    )  # search round with rewrite budget left ? grade : agent
    # Unconditional: a sufficient verdict adds nothing and the agent answers;
    # an insufficient one added grader guidance the agent must act on.
    graph.add_edge("grade", "agent")
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
        graph = _build_graph(
            _make_llm(cfg), build_tools(profile, trace), trace, question
        )
        config = {
            "configurable": {"thread_id": session_id or new_session_id()},
            # agent+tools+grade per search round, +1 for the closing agent
            # call — 3 supersteps per round bounds the graded worst case.
            "recursion_limit": 3 * max(1, max_steps) + 1,
        }

        prior_state = graph.get_state(config)
        prior_count = len((prior_state.values or {}).get("messages", []))

        exhausted = False
        try:
            # rewrites resets every turn: the budget is per question, but the
            # checkpointer would otherwise carry last turn's count forward.
            state = graph.invoke(
                {"messages": [HumanMessage(question)], "rewrites": 0}, config
            )
            messages = state["messages"]
        except GraphRecursionError:
            # Same recovery as the raw loop: one call with tools DISABLED
            # forces an answer from gathered evidence. The checkpointer has
            # every superstep up to the limit; write the answer back so the
            # thread stays coherent for the next turn.
            exhausted = True
            messages = graph.get_state(config).values["messages"]
            # The limit can hit right after the agent node emits tool_calls,
            # leaving them unanswered — a state the chat API rejects ("tool_calls
            # must be followed by tool messages"). Answer them synthetically so
            # both the forced call and the checkpointed thread stay well-formed.
            patch: list = []
            if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                patch = [
                    ToolMessage(
                        content="not executed: step budget exhausted",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                    for tc in messages[-1].tool_calls
                ]
            forced = _make_llm(cfg).invoke(
                [
                    SystemMessage(AGENT_SYSTEM_PROMPT),
                    *messages,
                    *patch,
                    HumanMessage(FORCED_ANSWER_PROMPT),
                ]
            )
            graph.update_state(config, {"messages": [*patch, forced]})
            messages = [*messages, *patch, forced]

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
