import { useState } from 'react'

const DEMO_ACCOUNTS = [
  { username: 'admin', password: 'admin', label: 'Quản trị viên', role: 'ADMIN', icon: '🛡️' },
  { username: 'hr01', password: 'hr', label: 'Nhân sự', role: 'HR', icon: '👥' },
  { username: 'ketoan01', password: 'ketoan', label: 'Kế toán', role: 'FINANCE', icon: '📊' },
]

const BLUE = '#1d4ed8'
const BLUE_DARK = '#1e3a8a'
const BLUE_LIGHT = '#eff6ff'
const BLUE_MID = '#3b82f6'

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [focused, setFocused] = useState(null)
  const [showPass, setShowPass] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const formData = new URLSearchParams()
      formData.append('username', username)
      formData.append('password', password)
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Đăng nhập thất bại')
      onLogin(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fieldStyle = (name) => ({
    width: '100%', padding: '11px 14px',
    background: 'white',
    border: `1.5px solid ${focused === name ? BLUE_MID : '#e2e8f0'}`,
    borderRadius: '10px', color: '#1e293b', fontSize: '14px',
    outline: 'none', transition: 'all 0.2s', boxSizing: 'border-box',
    boxShadow: focused === name ? `0 0 0 3px rgba(59,130,246,0.15)` : 'none',
    fontFamily: 'Inter, sans-serif',
  })

  return (
    <div className="login-bg">
      <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: '900px', padding: '24px', display: 'flex', gap: '48px', alignItems: 'center' }}>

        {/* Left branding panel */}
        <div style={{ flex: 1, display: 'none' }} className="hidden md:block">
        </div>
        <div style={{ flex: '0 0 auto', display: 'flex', flexDirection: 'column', gap: '20px', minWidth: '200px' }}>
          {/* University badge */}
          <div style={{
            background: `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
            borderRadius: '16px', padding: '28px 24px', color: 'white', textAlign: 'center',
            boxShadow: '0 12px 40px rgba(29,78,216,0.3)',
          }}>
            <div style={{ fontSize: '40px', marginBottom: '12px' }}>🎓</div>
            <div style={{ fontSize: '18px', fontWeight: '800', letterSpacing: '-0.3px', lineHeight: 1.2 }}>
              Đại học Mở
            </div>
            <div style={{ fontSize: '13px', opacity: 0.8, marginTop: '4px' }}>
              Knowledge Base AI
            </div>
            <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.2)' }}>
              <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '8px' }}>Phân quyền hệ thống</div>
              {DEMO_ACCOUNTS.map((a) => (
                <div key={a.role} style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '6px 0', fontSize: '12px'
                }}>
                  <span>{a.icon}</span>
                  <span style={{ opacity: 0.9 }}>{a.label}</span>
                  <span style={{
                    marginLeft: 'auto', background: 'rgba(255,255,255,0.2)',
                    padding: '2px 8px', borderRadius: '20px', fontSize: '10px', fontWeight: '600'
                  }}>{a.role}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Login card */}
        <div style={{
          flex: 1,
          background: 'white',
          borderRadius: '20px',
          padding: '40px',
          boxShadow: '0 20px 60px rgba(29,78,216,0.12), 0 4px 16px rgba(0,0,0,0.06)',
          border: '1px solid #e0eaff',
        }}>
          {/* Header */}
          <div style={{ marginBottom: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <div style={{
                width: '44px', height: '44px', borderRadius: '12px',
                background: `linear-gradient(135deg, ${BLUE}, ${BLUE_MID})`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '22px', boxShadow: '0 4px 12px rgba(29,78,216,0.3)',
              }}>🧠</div>
              <div>
                <h1 style={{ margin: 0, fontSize: '22px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.5px' }}>
                  RAG Enterprise
                </h1>
                <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>Hệ thống khai thác tri thức AI</p>
              </div>
            </div>
          </div>

          {error && (
            <div style={{
              background: '#fef2f2', border: '1px solid #fecaca',
              borderRadius: '10px', padding: '12px 14px', marginBottom: '20px',
              color: '#dc2626', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px'
            }}>
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>
                Tên đăng nhập
              </label>
              <input
                type="text" required value={username}
                onChange={(e) => setUsername(e.target.value)}
                onFocus={() => setFocused('user')} onBlur={() => setFocused(null)}
                placeholder="Nhập username..."
                style={fieldStyle('user')}
              />
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#374151', marginBottom: '6px' }}>
                Mật khẩu
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPass ? 'text' : 'password'} required value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setFocused('pass')} onBlur={() => setFocused(null)}
                  placeholder="••••••••"
                  style={{ ...fieldStyle('pass'), paddingRight: '44px' }}
                />
                <button type="button" onClick={() => setShowPass(!showPass)}
                  style={{
                    position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px',
                    color: '#94a3b8', padding: 0,
                  }}>
                  {showPass ? '🙈' : '👁️'}
                </button>
              </div>
            </div>

            <button type="submit" disabled={loading} style={{
              width: '100%', padding: '13px',
              background: loading ? '#93c5fd' : `linear-gradient(135deg, ${BLUE_DARK}, ${BLUE})`,
              border: 'none', borderRadius: '10px', color: 'white',
              fontSize: '15px', fontWeight: '700', cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s',
              boxShadow: loading ? 'none' : '0 4px 16px rgba(29,78,216,0.35)',
              fontFamily: 'Inter, sans-serif',
              letterSpacing: '0.01em',
            }}>
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <span style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.4)', borderTopColor: 'white', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite' }} />
                  Đang xác thực...
                </span>
              ) : '→  Đăng nhập'}
            </button>
          </form>

          {/* Quick-fill */}
          <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid #f1f5f9' }}>
            <p style={{ fontSize: '12px', color: '#94a3b8', textAlign: 'center', marginBottom: '10px', fontWeight: '500' }}>
              Đăng nhập nhanh (demo)
            </p>
            <div style={{ display: 'flex', gap: '8px' }}>
              {DEMO_ACCOUNTS.map((a) => (
                <button key={a.role} onClick={() => { setUsername(a.username); setPassword(a.password) }}
                  style={{
                    flex: 1, padding: '8px 6px',
                    background: BLUE_LIGHT, border: `1px solid #bfdbfe`,
                    borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s',
                    textAlign: 'center', fontFamily: 'Inter, sans-serif',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = '#dbeafe'; e.currentTarget.style.borderColor = BLUE_MID }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = BLUE_LIGHT; e.currentTarget.style.borderColor = '#bfdbfe' }}
                >
                  <div style={{ fontSize: '16px', marginBottom: '3px' }}>{a.icon}</div>
                  <div style={{ fontSize: '11px', fontWeight: '600', color: BLUE }}>{a.role}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
