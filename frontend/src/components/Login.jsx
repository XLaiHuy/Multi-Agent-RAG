import { useState } from 'react'

const DEMO_ACCOUNTS = [
  { username: 'admin', password: 'admin123', label: 'Quản trị viên', role: 'ADMIN', icon: '🛡️', color: 'from-purple-600 to-indigo-700' },
  { username: 'legal01', password: 'legal123', label: 'Pháp chế (Legal)', role: 'LEGAL', icon: '⚖️', color: 'from-indigo-600 to-blue-700' },
  { username: 'finance01', password: 'finance123', label: 'Tài chính (Finance)', role: 'FINANCE', icon: '💰', color: 'from-blue-600 to-cyan-700' },
  { username: 'hr01', password: 'hr123', label: 'Nhân sự (HR)', role: 'HR', icon: '👥', color: 'from-emerald-600 to-teal-700' },
  { username: 'user01', password: 'user123', label: 'Nhân viên (User)', role: 'USER', icon: '👤', color: 'from-slate-600 to-slate-700' },
]

export default function Login({ onLogin, apiUrl }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showPass, setShowPass] = useState(false)

  const baseUrl = apiUrl || 'http://localhost:8000'

  const executeLogin = async (userToLogin, passToLogin) => {
    setLoading(true)
    setError('')
    try {
      const formData = new URLSearchParams()
      formData.append('username', userToLogin)
      formData.append('password', passToLogin)

      const res = await fetch(`${baseUrl}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin.')
      }

      // data contains { access_token, token_type, user_info }
      onLogin(data.access_token, data.user_info)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!username || !password) return
    executeLogin(username, password)
  }

  const handleQuickLogin = (acc) => {
    setUsername(acc.username)
    setPassword(acc.password)
    executeLogin(acc.username, acc.password)
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4 selection:bg-indigo-500 selection:text-white">
      <div className="max-w-4xl w-full bg-slate-800 border border-slate-700 rounded-3xl shadow-2xl overflow-hidden grid grid-cols-1 md:grid-cols-12 animate-fadeIn">
        
        {/* Left Branding & Quick Accounts Strip */}
        <div className="md:col-span-5 bg-gradient-to-br from-indigo-950 via-slate-900 to-blue-950 p-8 flex flex-col justify-between border-b md:border-b-0 md:border-r border-slate-700 text-white">
          <div>
            <div className="w-14 h-14 rounded-2xl bg-white p-1.5 shadow-lg mb-4 ring-2 ring-indigo-400/30 flex items-center justify-center">
              <img src="/oulogo.png" alt="OU Logo" className="w-full h-full object-contain rounded-xl" />
            </div>
            <h2 className="text-xl font-black tracking-tight text-white">
              AI TRA CỨU LUẬT & HỢP ĐỒNG
            </h2>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Adaptive Multi-Agent RAG Platform dành cho thẩm định và rà soát hợp đồng doanh nghiệp.
            </p>

            <div className="mt-8 space-y-2">
              <div className="text-[11px] font-bold uppercase tracking-wider text-indigo-400 mb-2">
                ⚡ Đăng nhập nhanh tài khoản mẫu:
              </div>
              {DEMO_ACCOUNTS.map((a) => (
                <button
                  key={a.username}
                  type="button"
                  onClick={() => handleQuickLogin(a)}
                  className="w-full text-left p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 hover:border-indigo-500/50 transition flex items-center justify-between group active:scale-98"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-base">{a.icon}</span>
                    <div>
                      <div className="text-xs font-bold text-slate-200 group-hover:text-white">{a.label}</div>
                      <div className="text-[10px] text-slate-400 font-mono">{a.username}</div>
                    </div>
                  </div>
                  <span className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 font-semibold border border-indigo-500/30">
                    {a.role}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="text-[11px] text-slate-500 mt-6 pt-4 border-t border-slate-800">
            🔒 Phân quyền ACL đa Role & Bảo mật chống IDOR
          </div>
        </div>

        {/* Right Login Form */}
        <div className="md:col-span-7 p-8 md:p-10 bg-slate-850 flex flex-col justify-center text-white">
          <div className="mb-6">
            <h3 className="text-2xl font-bold text-white">Đăng Nhập</h3>
            <p className="text-xs text-slate-400 mt-1">
              Nhập thông tin xác thực để truy cập kho hợp đồng và các công cụ phân tích.
            </p>
          </div>

          {error && (
            <div className="mb-5 p-3.5 bg-red-500/10 border border-red-500/30 rounded-xl text-xs text-red-300 flex items-center gap-2">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                Tên đăng nhập (Username)
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Nhập admin, legal01, finance01, hr01..."
                required
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                Mật khẩu (Password)
              </label>
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Nhập mật khẩu..."
                  required
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white text-xs"
                >
                  {showPass ? 'Ẩn' : 'Hiện'}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !username || !password}
              className="w-full py-3.5 bg-gradient-to-r from-indigo-600 via-blue-600 to-indigo-600 hover:from-indigo-700 hover:to-blue-700 text-white font-bold text-sm rounded-xl shadow-lg transition disabled:opacity-50 flex items-center justify-center gap-2 mt-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Đang xác thực...</span>
                </>
              ) : (
                <>
                  <span>Xác Thực & Đăng Nhập</span>
                  <span>➔</span>
                </>
              )}
            </button>
          </form>

          {/* Quick Credential Hint Table */}
          <div className="mt-8 pt-6 border-t border-slate-800 text-[11px] text-slate-400">
            <div className="font-semibold text-slate-300 mb-1">Mật khẩu mặc định trong chế độ Development:</div>
            <div className="font-mono text-[10px] text-slate-500">
              admin: <code>admin123</code> | legal01: <code>legal123</code> | finance01: <code>finance123</code> | hr01: <code>hr123</code>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
