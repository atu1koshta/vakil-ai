import { useEffect, useRef, useState } from 'react';
import { agentAsk } from '../api.js';

const EXAMPLES = [
  'Which judgments are from the Bombay High Court?',
  'What did the court hold about reasonable restrictions in Chintaman Rao?',
  'Which cases from the 1950s discuss preventive detention?',
];

// One agent answer: the answer card plus the tool-call trace that produced it.
function AgentTurn({ result }) {
  return (
    <div className="space-y-2">
      <div className="rounded border border-slate-200">
        <div className="flex items-center gap-2 border-b border-slate-100 px-3 py-2 text-xs">
          <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-600">
            answer
          </span>
          <span className="font-mono text-slate-400">{result.model}</span>
          <span className="ml-auto shrink-0 text-slate-400">
            {result.iterations} LLM call{result.iterations === 1 ? '' : 's'}
          </span>
          {result.exhausted && (
            <span className="shrink-0 rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-800">
              forced after max steps
            </span>
          )}
        </div>
        <p className="whitespace-pre-wrap px-3 py-2 text-sm leading-relaxed text-slate-800">
          {result.answer}
        </p>
      </div>

      {result.steps.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-slate-500">
            Tool trace ({result.steps.length} call{result.steps.length === 1 ? '' : 's'})
          </p>
          {result.steps.map((s, i) => (
            <details
              key={i}
              className={`rounded border text-xs ${
                s.error ? 'border-red-200 bg-red-50' : 'border-slate-200'
              }`}
            >
              <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 hover:bg-slate-50">
                <span className="shrink-0 font-mono text-slate-400">#{i + 1}</span>
                <span
                  className={`shrink-0 rounded px-1.5 py-0.5 font-medium ${
                    s.error ? 'bg-red-100 text-red-800' : 'bg-sky-50 text-sky-700'
                  }`}
                >
                  {s.tool}
                </span>
                <span className="truncate font-mono text-slate-500">
                  {JSON.stringify(s.args)}
                </span>
                <span className="ml-auto shrink-0 font-mono text-slate-400">
                  {s.duration_ms}ms
                </span>
              </summary>
              <pre className="max-h-64 overflow-auto whitespace-pre-wrap border-t border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-800">
                {s.result_preview}
              </pre>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

// Chat thread over the agent: the backend keeps per-session history keyed by
// session_id, so follow-ups ("what about the earlier case?") resolve against
// prior turns. "New chat" drops the id — the next ask starts a fresh session.
export default function AgentPanel() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function run(q) {
    const text = (q ?? query).trim();
    if (!text || loading) return;
    setQuery('');
    setMessages((m) => [...m, { role: 'user', text }]);
    setLoading(true);
    try {
      const result = await agentAsk(text, { sessionId });
      setSessionId(result.session_id);
      setMessages((m) => [...m, { role: 'agent', result }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'error', text: e.message }]);
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([]);
    setSessionId(null);
    setQuery('');
  }

  return (
    <div className="flex h-full flex-col">
      <form
        className="flex gap-2 border-b border-slate-200 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask — the agent searches, filters and reads judgments on its own"
          className="min-w-0 flex-1 rounded border border-slate-300 px-3 py-1.5 text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-slate-800 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? 'Thinking…' : 'Ask'}
        </button>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={newChat}
            className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:border-slate-400"
          >
            New chat
          </button>
        )}
      </form>

      {messages.length === 0 && !loading && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-sm text-slate-400">
          <p>Agentic QA: the model picks its own tools — search, filter, read. Try:</p>
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => run(ex)}
                className="rounded border border-slate-200 bg-white px-3 py-1.5 text-left text-xs text-slate-600 hover:border-slate-400"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.length > 0 && (
        <div className="flex-1 space-y-3 overflow-y-auto p-3">
          {messages.map((m, i) => {
            if (m.role === 'user') {
              return (
                <div key={i} className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-slate-800 px-3 py-2 text-sm leading-relaxed text-white">
                    {m.text}
                  </p>
                </div>
              );
            }
            if (m.role === 'error') {
              return (
                <div
                  key={i}
                  className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700"
                >
                  {m.text}
                </div>
              );
            }
            return <AgentTurn key={i} result={m.result} />;
          })}
          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-500" />
              Agent looping over tools — a local model can take a minute or more.
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
