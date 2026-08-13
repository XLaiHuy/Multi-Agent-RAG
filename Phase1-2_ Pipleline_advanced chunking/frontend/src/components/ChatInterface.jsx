import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css' // Import highlight theme
import UploadModal from './UploadModal'

const BLUE = '#1d4ed8'
const BLUE_DARK = '#1e3a8a'
const BLUE_MID = '#3b82f6'
const BLUE_LIGHT = '#eff6ff'
const BLUE_100 = '#dbeafe'

const ROLE_COLORS = { admin: BLUE, HR: '#16a34a', Finance: '#d97706' }
const ROLE_ICONS = { admin: '🛡️', HR: '👥', Finance: '📊' }

const SUGGESTED = [
  'Chunking là gì và tại sao quan trọng trong RAG?',
  'DoRA khác LoRA như thế nào?',
  'Reranker hoạt động như thế nào?',
  'Giải thích Reciprocal Rank Fusion',
  'Dense Retrieval vs Sparse Retrieval?',
  'Chunk overlap có tác dụng gì?',
]

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => {})
}

function SourceModal({ source, index, onClose }) {
  const [copied, setCopied] = useState(false)
  if (!source) return null

  const handleCopy = () => {
    copyToClipboard(source.text || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Chuẩn hoá các đoạn nối ngữ cảnh [...liền kề...] thành divider đẹp mắt
  const formattedText = (source.text || '')
    .replace(/\[\.\.\.liền kề(?:\s*\([^)]*\))?\.\.\.\]/g, '\n\n---\n*🔗 **Đoạn nối ngữ cảnh liền kề***\n\n')

  const wordCount = formattedText.trim().split(/\s+/).filter(Boolean).length

  const docTitle = source.source && source.source !== 'unknown'
    ? source.source
    : (source.chunk_id ? source.chunk_id.split('_chunk_')[0] : 'Tài liệu tham khảo')

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000, padding: '20px'
    }} onClick={onClose}>
      <div style={{
        background: 'white', borderRadius: '18px', maxWidth: '680px', width: '100%',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        border: `1px solid ${BLUE_100}`, overflow: 'hidden',
        display: 'flex', flexDirection: 'column', maxHeight: '85vh'
      }} onClick={(e) => e.stopPropagation()}>
        
        {/* Header */}
        <div style={{
          padding: '16px 22px', background: `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
          color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '20px' }}>📖</span>
            <div>
              <div style={{ fontWeight: '700', fontSize: '15px' }}>Tài liệu tham khảo #{index}</div>
              <div style={{ fontSize: '12px', opacity: 0.85 }}>Trích xuất từ: {docTitle}</div>
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.2)', border: 'none', color: 'white',
            borderRadius: '50%', width: '30px', height: '30px', cursor: 'pointer',
            fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.2s'
          }}>&times;</button>
        </div>

        {/* Subheader / Metadata */}
        <div style={{
          padding: '10px 22px', background: '#f8fafc', borderBottom: `1px solid ${BLUE_100}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px', color: '#64748b'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <strong>Nguồn tài liệu:</strong>
            <span style={{ background: '#e0e7ff', color: '#1e3a8a', padding: '3px 8px', borderRadius: '6px', fontWeight: '600' }}>
              📁 {docTitle}
            </span>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <span>📝 {wordCount} từ</span>
            <span>📄 Định dạng Word</span>
          </div>
        </div>

        {/* Content (Rendered as Word Document) */}
        <div style={{ padding: '24px', overflowY: 'auto', background: '#f1f5f9', flexGrow: 1 }}>
          <div className="word-doc-container">
            {source.text ? (
              <ReactMarkdown rehypePlugins={[rehypeHighlight]} className="word-doc-view">
                {formattedText}
              </ReactMarkdown>
            ) : (
              <div style={{ color: '#94a3b8', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
                Không tìm thấy nội dung văn bản cho tài liệu này.
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '14px 22px', background: 'white', borderTop: `1px solid ${BLUE_100}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>
            Cơ sở tri thức RAG Doanh nghiệp
          </span>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={handleCopy} style={{
              padding: '8px 16px', background: 'white', border: `1px solid ${BLUE_100}`,
              borderRadius: '8px', color: copied ? '#16a34a' : BLUE, fontSize: '13px',
              fontWeight: '600', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
              display: 'flex', alignItems: 'center', gap: '6px'
            }}>
              {copied ? '✓ Đã sao chép' : '📋 Sao chép văn bản'}
            </button>
            <button onClick={onClose} style={{
              padding: '8px 18px', background: BLUE, color: 'white', border: 'none',
              borderRadius: '8px', fontSize: '13px', fontWeight: '600', cursor: 'pointer',
              fontFamily: 'Inter, sans-serif', boxShadow: '0 2px 8px rgba(29,78,216,0.25)'
            }}>
              Đóng
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Message({ msg, onCopy, onRegenerate, onSelectSource, isLast }) {
  const isUser = msg.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    copyToClipboard(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
    onCopy?.()
  }

  return (
    <div className="msg-animate" style={{ display: 'flex', flexDirection: isUser ? 'row-reverse' : 'row', gap: '10px', alignItems: 'flex-start', marginBottom: '16px' }}>
      {/* Avatar */}
      <div style={{
        width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0, marginTop: '2px',
        background: isUser ? `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})` : 'white',
        border: isUser ? 'none' : `1.5px solid ${BLUE_100}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '15px',
        boxShadow: isUser ? '0 2px 8px rgba(29,78,216,0.25)' : '0 1px 4px rgba(0,0,0,0.08)'
      }}>
        {isUser ? '👤' : '🧠'}
      </div>

      <div style={{ maxWidth: '78%', display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', overflow: 'hidden' }}>
        <div className={`message-bubble ${!isUser ? 'markdown-body' : ''}`} style={{
          padding: '12px 16px',
          borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
          background: isUser ? `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})` : 'white',
          border: isUser ? 'none' : `1px solid ${BLUE_100}`,
          color: isUser ? 'white' : '#1e293b',
          fontSize: '14px', lineHeight: '1.65',
          boxShadow: isUser ? '0 3px 12px rgba(29,78,216,0.2)' : '0 1px 6px rgba(0,0,0,0.07)',
          position: 'relative', overflowX: 'auto',
          width: '100%', boxSizing: 'border-box'
        }}>
          {isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
          ) : (
            <>
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                {msg.content}
              </ReactMarkdown>
              {msg.streaming && <span className="typing-cursor" />}
              {!msg.content && msg.status && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: BLUE_MID, fontSize: '13px' }}>
                  <span className="pulse-dot" style={{ width: '7px', height: '7px', borderRadius: '50%', background: BLUE_MID, display: 'inline-block', flexShrink: 0 }} />
                  {msg.status}
                </div>
              )}
            </>
          )}
        </div>

        {/* Action buttons (AI only, after done streaming) */}
        {!isUser && msg.content && !msg.streaming && (
          <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
            <button onClick={handleCopy} title="Sao chép" style={{
              padding: '4px 10px', border: `1px solid ${BLUE_100}`, borderRadius: '6px',
              background: 'white', color: copied ? '#16a34a' : '#64748b',
              fontSize: '12px', cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'Inter, sans-serif'
            }}>
              {copied ? '✓ Đã sao chép' : '📋 Sao chép'}
            </button>
            {isLast && onRegenerate && (
              <button onClick={onRegenerate} title="Hỏi lại" style={{
                padding: '4px 10px', border: `1px solid ${BLUE_100}`, borderRadius: '6px',
                background: 'white', color: '#64748b',
                fontSize: '12px', cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'Inter, sans-serif'
              }}>
                🔄 Hỏi lại
              </button>
            )}
          </div>
        )}

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
            {msg.sources.map((src, i) => (
              <button key={i} onClick={() => onSelectSource?.(src, i + 1)} style={{
                padding: '4px 12px', background: BLUE_LIGHT,
                border: `1px solid ${BLUE_100}`, borderRadius: '20px',
                fontSize: '12px', color: BLUE, fontWeight: '600',
                cursor: 'pointer', transition: 'all 0.15s', fontFamily: 'Inter, sans-serif',
                display: 'inline-flex', alignItems: 'center', gap: '5px',
                boxShadow: '0 1px 2px rgba(0,0,0,0.03)'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = BLUE_100 }}
              onMouseLeave={(e) => { e.currentTarget.style.background = BLUE_LIGHT }}
              >
                📖 Tài liệu tham khảo [{i + 1}]
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatInterface({ token, user, onLogout }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [conversations, setConversations] = useState([{ id: 1, title: 'Cuộc trò chuyện mới', messages: [] }])
  const [activeConvId, setActiveConvId] = useState(1)
  const [charCount, setCharCount] = useState(0)
  const [showUpload, setShowUpload] = useState(false)
  const [activeSource, setActiveSource] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const roleColor = ROLE_COLORS[user.role] || BLUE
  const roleIcon = ROLE_ICONS[user.role] || '👤'

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const newConversation = () => {
    const id = Date.now()
    setConversations(prev => [...prev, { id, title: 'Cuộc trò chuyện mới', messages: [] }])
    setActiveConvId(id)
    setMessages([])
  }

  const switchConversation = (conv) => {
    setActiveConvId(conv.id)
    setMessages(conv.messages)
  }

  const clearChat = () => {
    setMessages([])
    setConversations(prev => prev.map(c => c.id === activeConvId ? { ...c, messages: [] } : c))
  }

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || loading) return

    const userMsg = { role: 'user', content: query }
    const aiPlaceholder = { role: 'assistant', content: '', status: 'Đang định tuyến câu hỏi...', sources: [], streaming: true }
    
    // Gửi chat_history
    const chatHistory = messages.map(m => ({ role: m.role, content: m.content }))

    setMessages(prev => {
      const updated = [...prev, userMsg, aiPlaceholder]
      setConversations(convs => convs.map(c => c.id === activeConvId
        ? { ...c, title: query.slice(0, 40) + (query.length > 40 ? '...' : ''), messages: updated }
        : c
      ))
      return updated
    })
    setInput('')
    setCharCount(0)
    setLoading(true)

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ query, chat_history: chatHistory }),
      })
      if (res.status === 401) { onLogout(); return }
      if (res.status === 403) { throw new Error('Từ chối truy cập. Token có thể đã hết hạn.') }
      if (!res.ok) throw new Error('Server error')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.replace('data: ', '').trim()
          if (raw === '[DONE]') { setLoading(false); continue }
          try {
            const data = JSON.parse(raw)
            setMessages(prev => {
              const msgs = [...prev]
              const last = { ...msgs[msgs.length - 1] }
              if (data.message) last.status = data.message
              else if (data.sources) last.sources = data.sources
              else if (data.token) { last.content += data.token; last.status = null }
              msgs[msgs.length - 1] = last
              setConversations(convs => convs.map(c => c.id === activeConvId ? { ...c, messages: msgs } : c))
              return msgs
            })
          } catch (_) {}
        }
      }
    } catch (err) {
      if (err.message.includes('Từ chối')) {
          onLogout()
          return
      }
      setMessages(prev => {
        const msgs = [...prev]
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: '❌ Lỗi kết nối Backend. Vui lòng thử lại.', status: null, streaming: false }
        return msgs
      })
    } finally {
      setLoading(false)
      setMessages(prev => {
        const msgs = [...prev]
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], streaming: false }
        return msgs
      })
    }
  }, [loading, token, activeConvId, onLogout, messages])

  const lastUserQuery = messages.filter(m => m.role === 'user').at(-1)?.content

  const btnStyle = (active = false) => ({
    padding: '7px 12px', border: `1px solid ${active ? BLUE_MID : '#e2e8f0'}`,
    borderRadius: '8px', background: active ? BLUE_LIGHT : 'white',
    color: active ? BLUE : '#64748b', fontSize: '12px', fontWeight: '500',
    cursor: 'pointer', transition: 'all 0.15s', fontFamily: 'Inter, sans-serif',
  })

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#f8faff', overflow: 'hidden' }}>

      {/* Sidebar */}
      <div style={{
        width: sidebarOpen ? '260px' : '0px', overflow: 'hidden', flexShrink: 0,
        transition: 'width 0.3s ease', borderRight: `1px solid ${BLUE_100}`,
        background: 'white', display: 'flex', flexDirection: 'column',
        boxShadow: '2px 0 12px rgba(29,78,216,0.05)',
      }}>
        <div style={{ width: '260px', display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Logo */}
          <div style={{ padding: '16px', borderBottom: `1px solid ${BLUE_100}`, background: `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ fontSize: '24px' }}>🎓</div>
              <div>
                <div style={{ fontSize: '14px', fontWeight: '800', color: 'white' }}>RAG Enterprise</div>
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)' }}>Đại học Mở</div>
              </div>
            </div>
          </div>

          {/* User badge */}
          <div style={{ padding: '14px 16px', borderBottom: `1px solid ${BLUE_100}`, background: BLUE_LIGHT }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                width: '38px', height: '38px', borderRadius: '10px', flexShrink: 0,
                background: roleColor + '22', border: `1.5px solid ${roleColor}44`,
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px'
              }}>{roleIcon}</div>
              <div>
                <div style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>{user.full_name}</div>
                <div style={{
                  display: 'inline-block', padding: '1px 8px', borderRadius: '20px',
                  background: roleColor + '18', border: `1px solid ${roleColor}33`,
                  fontSize: '10px', color: roleColor, fontWeight: '700', marginTop: '2px'
                }}>{user.role.toUpperCase()}</div>
              </div>
            </div>
          </div>

          {/* New chat button */}
          <div style={{ padding: '12px 16px', borderBottom: `1px solid ${BLUE_100}` }}>
            <button onClick={newConversation} style={{
              width: '100%', padding: '9px 12px',
              background: `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
              border: 'none', borderRadius: '10px', color: 'white',
              fontSize: '13px', fontWeight: '600', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              boxShadow: '0 2px 8px rgba(29,78,216,0.25)', fontFamily: 'Inter, sans-serif',
              transition: 'all 0.2s',
            }}>
              ✏️ Cuộc trò chuyện mới
            </button>
          </div>

          {/* Conversations list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 8px' }}>
            <div style={{ fontSize: '10px', fontWeight: '700', color: '#94a3b8', letterSpacing: '0.08em', padding: '4px 8px 8px', textTransform: 'uppercase' }}>
              Lịch sử
            </div>
            {conversations.map(conv => (
              <button key={conv.id} onClick={() => switchConversation(conv)} style={{
                display: 'block', width: '100%', textAlign: 'left',
                padding: '9px 12px', marginBottom: '3px',
                background: conv.id === activeConvId ? BLUE_LIGHT : 'transparent',
                border: `1px solid ${conv.id === activeConvId ? BLUE_100 : 'transparent'}`,
                borderRadius: '9px', color: conv.id === activeConvId ? BLUE : '#475569',
                fontSize: '12px', cursor: 'pointer', transition: 'all 0.15s',
                fontFamily: 'Inter, sans-serif', fontWeight: conv.id === activeConvId ? '600' : '400',
              }}>
                💬 {conv.title}
              </button>
            ))}
          </div>

          {/* Suggested questions */}
          <div style={{ padding: '10px 8px', borderTop: `1px solid ${BLUE_100}` }}>
            <div style={{ fontSize: '10px', fontWeight: '700', color: '#94a3b8', letterSpacing: '0.08em', padding: '4px 8px 8px', textTransform: 'uppercase' }}>
              Gợi ý câu hỏi
            </div>
            <div style={{ maxHeight: '160px', overflowY: 'auto' }}>
              {SUGGESTED.map((q, i) => (
                <button key={i} onClick={() => sendMessage(q)} disabled={loading} style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '7px 10px', marginBottom: '3px',
                  background: 'transparent', border: '1px solid transparent',
                  borderRadius: '8px', color: '#475569',
                  fontSize: '12px', cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s', lineHeight: '1.4', fontFamily: 'Inter, sans-serif',
                }}
                  onMouseEnter={(e) => { if (!loading) { e.currentTarget.style.background = BLUE_LIGHT; e.currentTarget.style.color = BLUE; e.currentTarget.style.borderColor = BLUE_100 }}}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'transparent' }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Logout */}
          <div style={{ padding: '12px 16px', borderTop: `1px solid ${BLUE_100}` }}>
            <button onClick={onLogout} style={{
              width: '100%', padding: '9px',
              background: '#fef2f2', border: '1px solid #fecaca',
              borderRadius: '9px', color: '#dc2626', fontSize: '13px',
              cursor: 'pointer', transition: 'all 0.15s', fontFamily: 'Inter, sans-serif', fontWeight: '500',
            }}>
              ← Đăng xuất
            </button>
          </div>
        </div>
      </div>

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{
          height: '56px', display: 'flex', alignItems: 'center', padding: '0 20px', gap: '12px',
          background: 'white', borderBottom: `1px solid ${BLUE_100}`,
          boxShadow: '0 1px 8px rgba(29,78,216,0.07)',
        }}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{
            width: '34px', height: '34px', borderRadius: '8px', border: `1px solid ${BLUE_100}`,
            background: BLUE_LIGHT, cursor: 'pointer', color: BLUE, fontSize: '14px',
            display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s',
          }}>☰</button>

          <div style={{ width: '1px', height: '20px', background: BLUE_100 }} />
          <span style={{ fontSize: '14px', fontWeight: '700', color: '#1e293b' }}>Chat · Agentic RAG</span>

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Upload Button for Admin */}
            {user.role === 'admin' && (
              <button onClick={() => setShowUpload(true)} style={{
                padding: '6px 14px', background: '#10b981', color: 'white', border: 'none',
                borderRadius: '8px', fontWeight: '600', fontSize: '12px', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: '6px', boxShadow: '0 2px 6px rgba(16,185,129,0.3)',
                fontFamily: 'Inter, sans-serif'
              }}>
                📄 Tải tài liệu lên
              </button>
            )}
            
            {/* Clear chat */}
            {messages.length > 0 && (
              <button onClick={clearChat} style={btnStyle()}>
                🗑️ Xóa chat
              </button>
            )}
            {/* Model badge */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '5px 12px', background: BLUE_LIGHT,
              border: `1px solid ${BLUE_100}`, borderRadius: '20px',
            }}>
              <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 5px #22c55e' }} />
              <span style={{ fontSize: '12px', color: BLUE, fontWeight: '600' }}>Gemini Flash</span>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 20px', background: '#f8faff' }}>
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            {messages.length === 0 && (
              <div style={{ textAlign: 'center', paddingTop: '60px' }}>
                <div style={{
                  width: '72px', height: '72px', margin: '0 auto 20px',
                  background: `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
                  borderRadius: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '32px', boxShadow: '0 8px 24px rgba(29,78,216,0.25)',
                }}>🧠</div>
                <h2 style={{ margin: '0 0 8px', fontSize: '22px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.3px' }}>
                  Xin chào, {user.full_name}!
                </h2>
                <p style={{ margin: '0 0 32px', color: '#64748b', fontSize: '14px', lineHeight: 1.6 }}>
                  Tôi là trợ lý AI của <strong style={{ color: BLUE }}>Đại học Mở</strong>.<br/>
                  Quyền truy cập: <span style={{
                    display: 'inline-block', padding: '1px 10px', borderRadius: '20px',
                    background: roleColor + '18', border: `1px solid ${roleColor}33`,
                    color: roleColor, fontWeight: '700', fontSize: '13px'
                  }}>{user.role}</span>
                </p>
                {/* Quick-start grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', textAlign: 'left' }}>
                  {SUGGESTED.slice(0, 4).map((q, i) => (
                    <button key={i} onClick={() => sendMessage(q)} style={{
                      padding: '14px', background: 'white',
                      border: `1px solid ${BLUE_100}`, borderRadius: '12px',
                      color: '#374151', fontSize: '13px', cursor: 'pointer',
                      transition: 'all 0.2s', textAlign: 'left', lineHeight: '1.4',
                      fontFamily: 'Inter, sans-serif',
                      boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
                    }}
                      onMouseEnter={(e) => { e.currentTarget.style.borderColor = BLUE_MID; e.currentTarget.style.background = BLUE_LIGHT }}
                      onMouseLeave={(e) => { e.currentTarget.style.borderColor = BLUE_100; e.currentTarget.style.background = 'white' }}
                    >
                      <div style={{ fontSize: '18px', marginBottom: '6px' }}>{'🔍📚🔄📈'.charAt(i * 2)}{'🔍📚🔄📈'.charAt(i * 2 + 1)}</div>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <Message
                key={i} msg={msg}
                isLast={i === messages.length - 1 && msg.role === 'assistant'}
                onSelectSource={(src, index) => setActiveSource({ source: src, index })}
                onRegenerate={lastUserQuery ? () => {
                  setMessages(prev => prev.slice(0, -1))
                  setTimeout(() => sendMessage(lastUserQuery), 100)
                } : undefined}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div style={{ background: 'white', borderTop: `1px solid ${BLUE_100}`, padding: '14px 20px', boxShadow: '0 -1px 8px rgba(29,78,216,0.06)' }}>
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <form onSubmit={(e) => { e.preventDefault(); sendMessage(input) }} style={{ display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
              <div style={{ flex: 1, position: 'relative' }}>
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => { setInput(e.target.value); setCharCount(e.target.value.length) }}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input) } }}
                  placeholder="Hỏi về tài liệu của bạn... (Enter để gửi, Shift+Enter để xuống dòng)"
                  disabled={loading}
                  rows={1}
                  style={{
                    width: '100%', padding: '12px 16px', paddingRight: '60px',
                    background: BLUE_LIGHT, border: `1.5px solid ${input ? BLUE_MID : BLUE_100}`,
                    borderRadius: '14px', color: '#1e293b', fontSize: '14px', outline: 'none',
                    resize: 'none', lineHeight: '1.5', boxSizing: 'border-box',
                    transition: 'all 0.2s', fontFamily: 'Inter, sans-serif',
                    boxShadow: input ? `0 0 0 3px rgba(59,130,246,0.12)` : 'none',
                    maxHeight: '120px',
                  }}
                  onFocus={(e) => { e.target.style.borderColor = BLUE_MID; e.target.style.boxShadow = `0 0 0 3px rgba(59,130,246,0.15)` }}
                  onBlur={(e) => { e.target.style.borderColor = input ? BLUE_100 : BLUE_100; e.target.style.boxShadow = 'none' }}
                />
                {charCount > 0 && (
                  <span style={{ position: 'absolute', right: '12px', bottom: '10px', fontSize: '11px', color: '#94a3b8' }}>
                    {charCount}
                  </span>
                )}
              </div>
              <button type="submit" disabled={!input.trim() || loading} style={{
                width: '46px', height: '46px', flexShrink: 0,
                background: (!input.trim() || loading) ? '#e2e8f0' : `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
                border: 'none', borderRadius: '12px',
                cursor: (!input.trim() || loading) ? 'not-allowed' : 'pointer',
                fontSize: '18px', transition: 'all 0.2s',
                boxShadow: (!input.trim() || loading) ? 'none' : '0 4px 12px rgba(29,78,216,0.3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white',
              }}>
                {loading
                  ? <span style={{ width: '18px', height: '18px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite' }} />
                  : '↑'}
              </button>
            </form>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Enter để gửi · Shift+Enter xuống dòng</span>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Powered by Gemini Flash · ĐH Mở RAG</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Upload Modal */}
      {showUpload && <UploadModal token={token} onClose={() => setShowUpload(false)} />}
      
      {/* Source Detail Modal */}
      {activeSource && (
        <SourceModal
          source={activeSource.source}
          index={activeSource.index}
          onClose={() => setActiveSource(null)}
        />
      )}
      
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        /* Markdown styles */
        .markdown-body {
          font-family: inherit;
        }
        .markdown-body p { margin-top: 0; margin-bottom: 0.8em; }
        .markdown-body p:last-child { margin-bottom: 0; }
        .markdown-body code {
          background: rgba(0,0,0,0.06);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          font-size: 85%;
        }
        .markdown-body pre {
          background: #1e1e1e;
          color: #d4d4d4;
          padding: 12px;
          border-radius: 8px;
          overflow-x: auto;
          margin: 1em 0;
        }
        .markdown-body pre code {
          background: transparent;
          padding: 0;
          color: inherit;
        }
        .markdown-body ul, .markdown-body ol {
          margin-top: 0;
          margin-bottom: 0.8em;
          padding-left: 20px;
        }
        .markdown-body table {
          border-collapse: separate;
          border-spacing: 0;
          width: 100%;
          margin: 1.2em 0;
          border-radius: 10px;
          overflow: hidden;
          border: 1px solid #e2e8f0;
          box-shadow: 0 1px 6px rgba(0,0,0,0.04);
          font-size: 13px;
        }
        .markdown-body th, .markdown-body td {
          border-bottom: 1px solid #e2e8f0;
          border-right: 1px solid #e2e8f0;
          padding: 8px 14px;
          text-align: left;
        }
        .markdown-body th:last-child, .markdown-body td:last-child {
          border-right: none;
        }
        .markdown-body tr:last-child td {
          border-bottom: none;
        }
        .markdown-body th {
          background: #eff6ff;
          color: #1e3a8a;
          font-weight: 700;
          font-size: 12.5px;
          letter-spacing: 0.02em;
        }
        .markdown-body tr:nth-child(even) td {
          background: #f8fafc;
        }
        .markdown-body blockquote {
          border-left: 4px solid #3b82f6;
          background: #eff6ff;
          padding: 8px 14px;
          margin: 1em 0;
          border-radius: 0 8px 8px 0;
          color: #1e40af;
          font-size: 13px;
        }
        .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
          margin-top: 1em;
          margin-bottom: 0.5em;
          font-weight: 700;
          color: #0f172a;
        }
      `}</style>
    </div>
  )
}
