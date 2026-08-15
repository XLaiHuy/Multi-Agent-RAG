import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import CitationViewerModal from './CitationViewerModal'
import DebugPanel from './DebugPanel'

const SUGGESTED_QUESTIONS = [
  'What is the notice period required for termination without cause?',
  'What is the maximum aggregate liability cap under the agreement?',
  'Which jurisdiction governs dispute resolution and arbitration?',
  'What indemnification obligations are provided against IP infringement?',
]

export default function ChatInterface({ documents, token, apiUrl, user }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hello ${user?.full_name || 'there'}! I am your **Contract Intelligence Assistant**. You can ask detailed legal and operational questions about your uploaded agreements.`,
      citations: [],
    },
  ])
  const [inputQuery, setInputQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedDocId, setSelectedDocId] = useState('')
  const [activeCitation, setActiveCitation] = useState(null)
  const [latestStats, setLatestStats] = useState(null)
  const [showDebug, setShowDebug] = useState(false)

  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async (queryToSend) => {
    const q = (queryToSend || inputQuery).trim()
    if (!q || loading) return

    const userMsg = { role: 'user', content: q }
    setMessages((prev) => [...prev, userMsg])
    setInputQuery('')
    setLoading(true)

    try {
      const payload = {
        query: q,
        document_ids: selectedDocId ? [selectedDocId] : null,
      }

      const res = await fetch(`${apiUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Error getting response.')
      }

      const data = await res.json()

      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        verification_status: data.verification_status,
        stats: data.stats,
      }

      setMessages((prev) => [...prev, assistantMsg])
      if (data.stats) {
        setLatestStats(data.stats)
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Error:** ${err.message}`,
          citations: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50 relative">
      {/* Subheader: Scope & Debug Controls */}
      <div className="h-12 bg-white border-b border-slate-200 px-6 flex items-center justify-between text-xs shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 font-semibold">Scope:</span>
          <select
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            className="px-2.5 py-1 bg-slate-100 border border-slate-300 rounded-lg text-xs font-medium focus:ring-1 focus:ring-indigo-500 focus:outline-none"
          >
            <option value="">🔍 All Accessible Contracts ({documents.length})</option>
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                📄 {d.filename}
              </option>
            ))}
          </select>
        </div>

        {latestStats && (
          <button
            onClick={() => setShowDebug(!showDebug)}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition flex items-center gap-1 border border-slate-300"
          >
            <span>⚡ Telemetry</span>
            <span className="font-mono text-indigo-600 font-bold">
              {Math.round(latestStats.total_ms || 0)}ms
            </span>
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex gap-3.5 ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
          >
            {m.role === 'assistant' && (
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-700 to-blue-600 text-white flex items-center justify-center text-sm shrink-0 shadow-sm">
                ⚖️
              </div>
            )}

            <div
              className={`max-w-2xl rounded-2xl p-5 shadow-xs text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-xs font-medium'
                  : 'bg-white border border-slate-200 text-slate-800 rounded-bl-xs'
              }`}
            >
              <div className="prose prose-sm max-w-none prose-slate">
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>

              {/* Citations & Evidence Pills */}
              {m.citations?.length > 0 && (
                <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                    Evidence:
                  </span>
                  {m.citations.map((c, cIdx) => (
                    <button
                      key={cIdx}
                      onClick={() => setActiveCitation(c)}
                      className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition border border-indigo-200 flex items-center gap-1 active:scale-95 shadow-2xs"
                      title={c.supporting_text}
                    >
                      <span>📌</span> P.{c.page} {c.section_path?.[0] || 'Clause'}
                    </button>
                  ))}

                  {m.verification_status && (
                    <span
                      className={`ml-auto text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border ${
                        m.verification_status === 'grounded'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : 'bg-amber-50 text-amber-700 border-amber-200'
                      }`}
                    >
                      {m.verification_status}
                    </span>
                  )}
                </div>
              )}
            </div>

            {m.role === 'user' && (
              <div className="w-8 h-8 rounded-xl bg-slate-800 text-white flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
                {user?.username?.slice(0, 2).toUpperCase() || 'U'}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3.5 justify-start animate-pulse">
            <div className="w-8 h-8 rounded-xl bg-indigo-600 text-white flex items-center justify-center text-sm shrink-0">
              ⚖️
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-xs text-xs text-slate-500 flex items-center gap-2.5">
              <div className="w-3.5 h-3.5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
              <span>Evaluating retrieval confidence & synthesizing grounded answer...</span>
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Suggested Prompts (if only welcome message) */}
      {messages.length === 1 && (
        <div className="px-6 pb-2 max-w-4xl mx-auto w-full">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
            Suggested Contract Inquiries:
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {SUGGESTED_QUESTIONS.map((s, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(s)}
                className="text-left px-3.5 py-2 rounded-xl bg-white border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/40 text-xs text-slate-700 transition font-medium truncate"
              >
                💡 {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat Input Bar */}
      <div className="p-4 bg-white border-t border-slate-200 shrink-0">
        <div className="max-w-4xl mx-auto flex items-center gap-2">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask anything about the contract clauses, liability, termination..."
            className="flex-1 px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition"
          />
          <button
            onClick={() => handleSend()}
            disabled={loading || !inputQuery.trim()}
            className="px-5 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-sm transition disabled:opacity-40 flex items-center gap-1.5 shrink-0"
          >
            <span>Ask</span>
            <span>➤</span>
          </button>
        </div>
      </div>

      {/* Citation Modal */}
      {activeCitation && (
        <CitationViewerModal citation={activeCitation} onClose={() => setActiveCitation(null)} />
      )}

      {/* Observability Telemetry Drawer */}
      {showDebug && latestStats && (
        <DebugPanel stats={latestStats} onClose={() => setShowDebug(false)} />
      )}
    </div>
  )
}
