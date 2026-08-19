import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import CitationViewerModal from './CitationViewerModal'
import DebugPanel from './DebugPanel'
import { createChatStream } from '../api/chat'

const DEMO_QUESTIONS = [
  {
    title: 'Bồi thường khi chấm dứt hợp đồng dịch vụ / lao động trái luật',
    query: 'Quy định bồi thường và nghĩa vụ khi chấm dứt hợp đồng như thế nào?',
  },
  {
    title: 'Quy định mức phạt vi phạm vượt quá 8% theo Luật Thương mại Việt Nam',
    query: 'Mức phạt vi phạm hợp đồng được quy định là bao nhiêu phần trăm?',
  },
  {
    title: 'Các trường hợp hợp đồng, điều khoản đặt cọc và thanh toán bị vô hiệu',
    query: 'Có điều khoản nào trong hợp đồng có nguy cơ bị tuyên vô hiệu không?',
  },
  {
    title: 'Nghĩa vụ bảo mật thông tin và bảo vệ dữ liệu cá nhân theo Nghị định 13',
    query: 'Hợp đồng quy định trách nhiệm bảo mật thông tin và bảo vệ dữ liệu cá nhân ra sao?',
  },
  {
    title: 'Mức trần bồi thường thiệt hại tối đa và miễn trừ trách nhiệm As-Is',
    query: 'Mức trần bồi thường thiệt hại tối đa (liability cap) là bao nhiêu?',
  },
]

export default function ChatInterface({
  documents = [],
  token: _token,
  apiUrl: _apiUrl,
  user,
  onLogout,
  onOpenUpload,
  onOpenWorkspace,
  resetSignal = 0,
}) {
  const storageKey = `chat_history_${user?.tenant_id || 'default'}_${user?.username || 'user'}`

  const getInitialMessages = () => {
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) return parsed
      }
    } catch (e) {
      console.warn('Failed to parse saved chat history:', e)
    }
    return [
      {
        role: 'assistant',
        content: `Xin chào **${user?.full_name || user?.username || 'quý khách'}**! Tôi là **Trợ lý AI Tra cứu & Phân tích Hợp đồng Doanh nghiệp**.\n\nHãy nhập câu hỏi hoặc chọn các câu hỏi mẫu bên dưới để tra cứu ngay với căn cứ trích dẫn chuẩn xác.`,
        citations: [],
      },
    ]
  }

  const [messages, setMessages] = useState(getInitialMessages)
  const [inputQuery, setInputQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamStage, setStreamStage] = useState('')
  const [selectedDocId, setSelectedDocId] = useState('')
  const [activeCitation, setActiveCitation] = useState(null)
  const [latestStats, setLatestStats] = useState(null)
  const [showDebug, setShowDebug] = useState(false)
  const [copiedIdx, setCopiedIdx] = useState(null)
  const [queryMode, setQueryMode] = useState('pro') // 'pro' | 'fast'

  const chatEndRef = useRef(null)

  // Reset chat when resetSignal triggers
  useEffect(() => {
    if (resetSignal > 0) {
      handleClearChat()
    }
  }, [resetSignal])

  // Save conversation whenever messages change
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(messages))
    } catch (e) {
      console.warn('Failed to save chat history to localStorage:', e)
    }
  }, [messages, storageKey])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleClearChat = () => {
    const defaultMsg = [
      {
        role: 'assistant',
        content: `Đã khởi tạo phiên làm việc mới. Bạn muốn tra cứu hay thẩm định điều khoản hợp đồng nào?`,
        citations: [],
      },
    ]
    setMessages(defaultMsg)
    setLatestStats(null)
  }

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 2000)
  }

  const handleSend = async (queryToSend) => {
    const q = (queryToSend || inputQuery).trim()
    if (!q || loading) return

    const userMsg = { role: 'user', content: q }
    setMessages((prev) => [...prev, userMsg])
    setInputQuery('')
    setLoading(true)
    setStreamStage('Đang kết nối pipeline pháp lý...')

    const payload = {
      query: q,
      document_ids: selectedDocId ? [selectedDocId] : null,
      mode: queryMode,
    }

    try {
      createChatStream(
        payload,
        (eventType, eventData) => {
          if (eventType === 'stage') {
            setStreamStage(eventData.message || 'Đang xử lý...')
          } else if (eventType === 'final') {
            const assistantMsg = {
              role: 'assistant',
              content: eventData.answer,
              citations: eventData.citations || [],
              verification_status: eventData.verification_status,
              stats: eventData.stats,
            }
            setMessages((prev) => [...prev, assistantMsg])
            if (eventData.stats) {
              setLatestStats(eventData.stats)
            }
            setLoading(false)
            setStreamStage('')
          }
        },
        (err) => {
          let content = `⚠️ **Lỗi:** ${err.message}`
          let isSessionExpired = false

          if (err.message === 'SESSION_EXPIRED' || err.message.toLowerCase().includes('token has expired') || err.message.includes('401')) {
            content = `⚠️ **Phiên làm việc đã hết hạn:** Mã xác thực đăng nhập đã hết hạn. Vui lòng đăng nhập lại để tiếp tục tra cứu.`
            isSessionExpired = true
          }

          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content,
              citations: [],
              isSessionExpired,
            },
          ])
          setLoading(false)
          setStreamStage('')
        },
        () => {
          setLoading(false)
          setStreamStage('')
        }
      )
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Lỗi:** ${err.message}`,
          citations: [],
        },
      ])
      setLoading(false)
      setStreamStage('')
    }
  }

  const isInitialState = messages.length <= 1

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-white relative">
      {/* 1. Subheader Controls Bar */}
      <div className="h-12 bg-white/90 backdrop-blur-sm border-b border-slate-100 px-6 flex items-center justify-between text-xs shrink-0 shadow-2xs">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 font-semibold flex items-center gap-1">
            <span>🎯</span> Phạm vi tra cứu:
          </span>
          <select
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            className="px-2.5 py-1 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-xs font-medium text-slate-800 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 focus:outline-none transition cursor-pointer max-w-[260px] truncate"
          >
            <option value="">🔍 Toàn bộ kho hợp đồng ({documents.length})</option>
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                📄 {d.filename}
              </option>
            ))}
          </select>

          {documents.length === 0 && (
            <button
              onClick={onOpenUpload}
              className="text-blue-600 hover:text-blue-800 font-bold ml-2 underline text-xs cursor-pointer"
            >
              + Tải hợp đồng mẫu lên ngay
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          {!isInitialState && (
            <button
              onClick={handleClearChat}
              className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 font-medium transition flex items-center gap-1 border border-slate-200 cursor-pointer"
              title="Khởi tạo phiên trò chuyện mới"
            >
              <span>🔄</span> Cuộc trò chuyện mới
            </button>
          )}

          {latestStats && (
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="px-2.5 py-1 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-700 font-medium transition flex items-center gap-1.5 border border-blue-200/80 cursor-pointer"
              title="Xem thông số Telemetry RAG"
            >
              <span>⚡ Telemetry:</span>
              <span className="font-mono font-bold text-blue-900">
                {Math.round(latestStats.total_ms || 0)}ms
              </span>
            </button>
          )}
        </div>
      </div>

      {/* 2. Main Workspace Body */}
      {isInitialState ? (
        /* HERO WELCOME SCREEN */
        <div className="flex-1 overflow-y-auto px-6 py-10 flex flex-col items-center justify-center max-w-3xl mx-auto w-full animate-fadeIn">
          {/* Official OU Logo */}
          <div className="w-20 h-20 rounded-3xl bg-white p-2.5 shadow-xl shadow-blue-500/10 mb-5 ring-4 ring-blue-50/90 border border-slate-200/90 flex items-center justify-center">
            <img src="/oulogo.png" alt="OU Logo" className="w-full h-full object-contain" />
          </div>

          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 text-center tracking-tight mb-2">
            AI Tra cứu Hợp đồng có thể hỗ trợ gì cho bạn?
          </h2>
          <p className="text-sm text-slate-500 text-center max-w-lg mb-8">
            Hỏi đáp điều khoản, tra cứu nghĩa vụ và bóc tách rủi ro pháp lý với căn cứ trích dẫn chuẩn xác từ văn bản gốc.
          </p>

          {/* Floating Central Input Card */}
          <div className="w-full bg-white rounded-2xl border-2 border-slate-200/90 hover:border-blue-300 focus-within:border-blue-600 focus-within:ring-4 focus-within:ring-blue-500/10 shadow-md shadow-slate-200/50 transition-all duration-200 p-4 mb-8">
            <textarea
              rows={3}
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder={
                documents.length === 0
                  ? 'Tải lên tài liệu hợp đồng hoặc nhập câu hỏi pháp lý của bạn tại đây...'
                  : 'Nhập câu hỏi về điều khoản hợp đồng, mức bồi thường, phạt vi phạm, chấm dứt...'
              }
              className="w-full bg-transparent border-none text-[15px] text-slate-800 placeholder-slate-400 focus:outline-none resize-none p-1 leading-relaxed"
            />

            <div className="flex items-center justify-between pt-3 border-t border-slate-100 mt-2">
              <div className="flex items-center gap-2.5">
                {/* Attach File Shortcut */}
                <button
                  onClick={onOpenUpload}
                  className="px-3 py-1.5 rounded-xl text-slate-600 hover:text-blue-700 hover:bg-blue-50 bg-slate-50 border border-slate-200/80 transition-all duration-150 text-xs font-semibold flex items-center gap-1.5 cursor-pointer"
                  title="Tải lên tệp hợp đồng (.txt, .pdf, .docx)"
                >
                  <span>📎</span>
                  <span>Đính kèm tệp</span>
                </button>

                {/* Mode Selector */}
                <button
                  onClick={() => setQueryMode(queryMode === 'pro' ? 'fast' : 'pro')}
                  className="px-3 py-1.5 rounded-xl text-xs font-semibold text-blue-700 bg-blue-50 hover:bg-blue-100 transition-all duration-150 flex items-center gap-1 cursor-pointer border border-blue-200/80"
                >
                  <span>✨ {queryMode === 'pro' ? 'Chuyên nghiệp (Safe Multi-Agent)' : 'Phân tích nhanh'}</span>
                  <span className="text-[10px] text-blue-500">⌄</span>
                </button>
              </div>

              <button
                onClick={() => handleSend()}
                disabled={loading || !inputQuery.trim()}
                className="px-4 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white flex items-center gap-1.5 text-xs font-bold shadow-sm shadow-blue-500/20 transition-all duration-150 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer active:scale-95"
              >
                <span>Gửi câu hỏi</span>
                <span className="text-xs">➤</span>
              </button>
            </div>
          </div>

          {/* Suggested Prompts with 'Demo' tag */}
          <div className="w-full space-y-2.5">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 px-1 flex items-center justify-between">
              <span>💡 Tình huống pháp lý & hợp đồng mẫu:</span>
              <span className="text-[11px] font-medium text-slate-400">Bấm để tra cứu ngay</span>
            </div>

            <div className="space-y-2.5">
              {DEMO_QUESTIONS.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q.query)}
                  className="w-full flex items-center justify-between p-4 rounded-xl bg-white hover:bg-blue-50/50 border border-slate-200 hover:border-blue-300 text-left transition-all duration-200 group cursor-pointer shadow-2xs hover:shadow-xs hover:-translate-y-0.5"
                >
                  <div className="flex items-center gap-3 truncate pr-3">
                    <span className="w-7 h-7 rounded-lg bg-slate-100 text-slate-500 group-hover:bg-blue-100 group-hover:text-blue-700 transition-colors duration-150 flex items-center justify-center text-xs shrink-0 font-bold">
                      §
                    </span>
                    <span className="text-sm font-semibold text-slate-700 group-hover:text-blue-900 truncate">
                      {q.title}
                    </span>
                  </div>
                  <span className="px-2.5 py-1 rounded-lg text-[11px] font-black uppercase tracking-wider bg-blue-100/80 text-blue-700 border border-blue-200/80 shrink-0 shadow-2xs group-hover:bg-blue-600 group-hover:text-white transition-colors duration-150">
                    DEMO
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* ACTIVE CHAT THREAD */
        <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex gap-3.5 ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn group`}
            >
              {m.role === 'assistant' && (
                <div className="w-10 h-10 rounded-2xl bg-white border border-slate-200/90 p-1 flex items-center justify-center text-sm shrink-0 shadow-xs ring-2 ring-blue-50">
                  <img src="/oulogo.png" alt="OU Legal AI" className="w-full h-full object-contain rounded-lg" />
                </div>
              )}

              <div
                className={`max-w-2xl rounded-2xl p-5 shadow-xs text-sm leading-relaxed relative ${
                  m.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-xs font-medium'
                    : 'bg-white border border-slate-200/90 text-slate-800 rounded-bl-xs'
                }`}
              >
                <div className="prose prose-sm max-w-none prose-slate">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>

                {/* Session Expired Action */}
                {m.isSessionExpired && onLogout && (
                  <div className="mt-4 pt-3 border-t border-red-100 flex items-center gap-2">
                    <button
                      onClick={onLogout}
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white font-bold text-xs rounded-xl shadow-xs transition flex items-center gap-1.5 cursor-pointer"
                    >
                      <span>🔑</span> Đăng nhập lại
                    </button>
                  </div>
                )}

                {/* Citations & Evidence Pills */}
                {m.citations?.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                      Căn cứ trích dẫn:
                    </span>
                    {m.citations.map((c, cIdx) => (
                      <div key={cIdx} className="flex items-center gap-1">
                        <button
                          onClick={() => setActiveCitation(c)}
                          className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 transition border border-blue-200/80 flex items-center gap-1 active:scale-95 shadow-2xs cursor-pointer"
                          title={c.supporting_text}
                        >
                          <span>📌</span> Trang {c.page} {c.section_path?.[0] || 'Điều khoản'}
                        </button>
                        {onOpenWorkspace && (
                          <button
                            onClick={() => onOpenWorkspace(c)}
                            className="px-1.5 py-1 text-[10px] rounded-lg bg-slate-100 hover:bg-blue-100 text-slate-600 hover:text-blue-700 transition cursor-pointer"
                            title="Mở tài liệu đối soát PDF"
                          >
                            🔍 PDF
                          </button>
                        )}
                      </div>
                    ))}

                    {m.verification_status && (
                      <span
                        className={`ml-auto text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border ${
                          m.verification_status === 'grounded'
                            ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                            : 'bg-amber-50 text-amber-700 border-amber-200'
                        }`}
                      >
                        {m.verification_status === 'grounded' ? 'Đã xác minh' : m.verification_status}
                      </span>
                    )}
                  </div>
                )}

                {/* Copy message button */}
                {m.role === 'assistant' && !m.isSessionExpired && (
                  <div className="mt-2 pt-2 border-t border-slate-50 flex justify-end">
                    <button
                      onClick={() => handleCopy(m.content, idx)}
                      className="text-[11px] text-slate-400 hover:text-slate-700 font-medium transition flex items-center gap-1 opacity-60 group-hover:opacity-100 cursor-pointer"
                    >
                      {copiedIdx === idx ? (
                        <span className="text-emerald-600 font-bold">✓ Đã sao chép</span>
                      ) : (
                        <>
                          <span>📋</span> Sao chép câu trả lời
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>

              {m.role === 'user' && (
                <div className="w-9 h-9 rounded-2xl bg-slate-800 text-white flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
                  {user?.username?.slice(0, 2).toUpperCase() || 'AD'}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3.5 justify-start animate-pulse">
              <div className="w-10 h-10 rounded-2xl bg-white border border-slate-200/90 p-1 flex items-center justify-center text-sm shrink-0 shadow-xs ring-2 ring-blue-50">
                <img src="/oulogo.png" alt="OU Legal AI" className="w-full h-full object-contain rounded-lg" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-xs text-xs text-slate-600 flex items-center gap-2.5">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <span>{streamStage || 'Đang tra cứu kho điều khoản & tổng hợp giải đáp pháp lý...'}</span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      )}

      {/* 3. Pinned Bottom Input Bar when active in chat */}
      {!isInitialState && (
        <div className="p-4 bg-white/95 backdrop-blur-sm border-t border-slate-200/80 shrink-0 shadow-lg shadow-slate-200/50">
          <div className="max-w-4xl mx-auto flex items-center gap-2.5">
            <button
              onClick={onOpenUpload}
              className="p-3 rounded-xl bg-slate-100 hover:bg-blue-50 text-slate-600 hover:text-blue-600 transition text-sm shrink-0 cursor-pointer"
              title="Đính kèm thêm hợp đồng"
            >
              📎
            </button>
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Hỏi bất kỳ điều gì về điều khoản, mức trần bồi thường, phạt vi phạm, thời hạn..."
              className="flex-1 px-4 py-3 bg-slate-50 border border-slate-300/80 rounded-xl text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 focus:bg-white transition"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !inputQuery.trim()}
              className="px-5 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl shadow-sm shadow-blue-500/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 shrink-0 cursor-pointer active:scale-95"
            >
              <span>Gửi</span>
              <span className="text-xs">➤</span>
            </button>
          </div>
        </div>
      )}

      {/* Citation Modal */}
      {activeCitation && (
        <CitationViewerModal
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
          onOpenWorkspace={onOpenWorkspace}
        />
      )}

      {/* Observability Telemetry Drawer */}
      {showDebug && latestStats && (
        <DebugPanel stats={latestStats} onClose={() => setShowDebug(false)} />
      )}
    </div>
  )
}
