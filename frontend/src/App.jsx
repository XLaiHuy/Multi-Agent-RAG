import { useState, useEffect, useRef } from 'react'
import ChatInterface from './components/ChatInterface'
import ContractCompareView from './components/ContractCompareView'
import RiskReviewDashboard from './components/RiskReviewDashboard'
import UploadModal from './components/UploadModal'
import Login from './components/Login'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '')
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user') || 'null'))
  const [activeTab, setActiveTab] = useState('qa') // 'qa' | 'compare' | 'risk' | 'library'
  const [documents, setDocuments] = useState([])
  const [showUpload, setShowUpload] = useState(false)
  const [docSearchQuery, setDocSearchQuery] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [chatResetSignal, setChatResetSignal] = useState(0)

  const fetchDocuments = async () => {
    if (!token) return
    try {
      const res = await fetch(`${API_URL}/api/v1/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setDocuments(data)
      } else if (res.status === 401) {
        handleLogout()
      }
    } catch (err) {
      console.error('Error fetching documents:', err)
    }
  }

  useEffect(() => {
    if (token) {
      fetchDocuments()
    }
  }, [token])

  const handleLogin = (accessToken, userInfo) => {
    setToken(accessToken)
    setUser(userInfo)
    localStorage.setItem('token', accessToken)
    localStorage.setItem('user', JSON.stringify(userInfo))
  }

  const handleLogout = () => {
    setToken('')
    setUser(null)
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  }

  const handleNewChat = () => {
    setActiveTab('qa')
    setChatResetSignal((prev) => prev + 1)
  }

  if (!token) {
    return <Login onLogin={handleLogin} apiUrl={API_URL} />
  }

  const filteredDocs = documents.filter((d) =>
    (d.filename || '').toLowerCase().includes(docSearchQuery.toLowerCase()) ||
    (d.file_type || '').toLowerCase().includes(docSearchQuery.toLowerCase())
  )

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans antialiased overflow-hidden selection:bg-indigo-500 selection:text-white">
      {/* 1. LEFT SIDEBAR (Spacious & Clean Legal SaaS Design) */}
      <aside
        className={`${
          sidebarOpen ? 'w-72' : 'w-20'
        } bg-white border-r border-slate-200/90 flex flex-col justify-between transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] z-40 shrink-0 select-none shadow-xs`}
      >
        {/* Top Header & Brand */}
        <div className="flex flex-col">
          {/* Logo Brand */}
          <div className="h-16 px-4 flex items-center justify-between border-b border-slate-100">
            <div
              className="flex items-center gap-3 overflow-hidden cursor-pointer group"
              onClick={() => setActiveTab('qa')}
            >
              <div className="w-11 h-11 rounded-xl bg-white p-1 border border-slate-200/90 shadow-sm flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform duration-200 ring-2 ring-blue-50">
                <img src="/oulogo.png" alt="OU Logo" className="w-full h-full object-contain rounded-lg" />
              </div>
              {sidebarOpen && (
                <div className="flex flex-col truncate">
                  <span className="font-extrabold text-[15px] tracking-tight text-slate-900 flex items-center gap-1 leading-tight">
                    AI TRA CỨU <span className="text-blue-600 font-black">LUẬT</span>
                  </span>
                  <span className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mt-0.5">
                    Safe Contract Intelligence
                  </span>
                </div>
              )}
            </div>

            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors duration-150 text-sm cursor-pointer"
              title={sidebarOpen ? 'Thu gọn sidebar' : 'Mở rộng sidebar'}
            >
              {sidebarOpen ? '◀' : '▶'}
            </button>
          </div>

          {/* Navigation Menu */}
          <div className="p-3.5 space-y-6 overflow-y-auto">
            {/* Group 1: Trợ lý AI */}
            <div>
              {sidebarOpen && (
                <div className="px-3 mb-2.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Trợ lý AI Pháp lý
                </div>
              )}

              <div className="space-y-1.5">
                {/* Tab: Hỏi đáp */}
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setActiveTab('qa')}
                    className={`flex-1 flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                      activeTab === 'qa'
                        ? 'bg-blue-50/90 text-blue-700 font-bold shadow-2xs border border-blue-100/90'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                    }`}
                  >
                    <span className="text-base shrink-0">💬</span>
                    {sidebarOpen && <span className="truncate">Hỏi đáp Pháp lý</span>}
                  </button>
                  {sidebarOpen && (
                    <button
                      onClick={handleNewChat}
                      className="p-2 rounded-xl text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-colors duration-150 text-sm font-bold cursor-pointer"
                      title="Cuộc trò chuyện mới (+)"
                    >
                      ➕
                    </button>
                  )}
                </div>

                {/* Tab: Review Hợp đồng */}
                <button
                  onClick={() => setActiveTab('risk')}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                    activeTab === 'risk'
                      ? 'bg-blue-50/90 text-blue-700 font-bold shadow-2xs border border-blue-100/90'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                  }`}
                >
                  <div className="flex items-center gap-3 truncate">
                    <span className="text-base shrink-0">🛡️</span>
                    {sidebarOpen && <span className="truncate">Review Hợp đồng</span>}
                  </div>
                  {sidebarOpen && (
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 uppercase tracking-wide">
                      HOT
                    </span>
                  )}
                </button>

                {/* Tab: So sánh Hợp đồng */}
                <button
                  onClick={() => setActiveTab('compare')}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                    activeTab === 'compare'
                      ? 'bg-blue-50/90 text-blue-700 font-bold shadow-2xs border border-blue-100/90'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                  }`}
                >
                  <div className="flex items-center gap-3 truncate">
                    <span className="text-base shrink-0">⚖️</span>
                    {sidebarOpen && <span className="truncate">So sánh Hợp đồng</span>}
                  </div>
                  {sidebarOpen && (
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded-md bg-blue-100 text-blue-800 uppercase tracking-wide">
                      Mới
                    </span>
                  )}
                </button>
              </div>
            </div>

            {/* Group 2: Dữ liệu & Kho hợp đồng */}
            <div>
              {sidebarOpen && (
                <div className="px-3 mb-2.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Quản lý Dữ liệu
                </div>
              )}

              <div className="space-y-1.5">
                <button
                  onClick={() => setActiveTab('library')}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                    activeTab === 'library'
                      ? 'bg-blue-50/90 text-blue-700 font-bold shadow-2xs border border-blue-100/90'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                  }`}
                >
                  <div className="flex items-center gap-3 truncate">
                    <span className="text-base shrink-0">📚</span>
                    {sidebarOpen && <span className="truncate">Kho Hợp đồng</span>}
                  </div>
                  {sidebarOpen && (
                    <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                      {documents.length}
                    </span>
                  )}
                </button>

                <button
                  onClick={() => setShowUpload(true)}
                  className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold text-slate-600 hover:text-blue-700 hover:bg-blue-50/60 transition-colors duration-150 cursor-pointer"
                >
                  <span className="text-base shrink-0">📤</span>
                  {sidebarOpen && <span className="truncate">+ Tải lên hợp đồng</span>}
                </button>
              </div>
            </div>

            {/* Group 3: Lịch sử hỏi đáp */}
            {sidebarOpen && (
              <div>
                <div className="px-3 mb-2.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Lịch sử hỏi đáp
                </div>
                <div className="px-3 py-2 text-xs text-slate-500 bg-slate-50 rounded-xl border border-slate-100 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0"></span>
                  <span className="truncate font-medium">Phiên hiện tại đang hoạt động</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom User Info & Prominent Logout (Clear & Obvious Logout Button) */}
        <div className="p-3 border-t border-slate-100 bg-slate-50/60 space-y-2">
          {/* User Profile Card */}
          <div className="flex items-center justify-between gap-2.5 p-2.5 rounded-xl bg-white border border-slate-200/90 shadow-2xs">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center font-bold text-xs shadow-xs shrink-0 ring-2 ring-blue-100">
                {user?.username?.slice(0, 2).toUpperCase() || 'AD'}
              </div>
              {sidebarOpen && (
                <div className="flex flex-col truncate">
                  <span className="font-bold text-xs text-slate-900 truncate leading-tight">
                    {user?.full_name || user?.username || 'Chuyên viên pháp lý'}
                  </span>
                  <span className="text-[10px] text-blue-600 font-bold uppercase tracking-wider mt-0.5">
                    Gói Doanh Nghiệp
                  </span>
                </div>
              )}
            </div>

            {/* Red Quick Logout Button */}
            <button
              onClick={handleLogout}
              className="px-2.5 py-1.5 rounded-lg bg-red-50 hover:bg-red-600 text-red-600 hover:text-white font-bold transition-all duration-150 text-xs flex items-center gap-1 border border-red-200/80 hover:border-red-600 shadow-2xs cursor-pointer active:scale-95 shrink-0"
              title="Đăng xuất khỏi hệ thống"
            >
              <span>🚪</span>
              {sidebarOpen && <span>Thoát</span>}
            </button>
          </div>

          {/* Full Logout Bar when Sidebar Open */}
          {sidebarOpen && (
            <button
              onClick={handleLogout}
              className="w-full py-2 px-3 rounded-xl bg-slate-100 hover:bg-red-50 text-slate-600 hover:text-red-700 font-bold text-xs transition-all duration-150 flex items-center justify-center gap-2 border border-slate-200/80 hover:border-red-200 cursor-pointer"
            >
              <span>🚪</span>
              <span>Đăng xuất tài khoản</span>
            </button>
          )}
        </div>
      </aside>

      {/* 2. MAIN WORKSPACE */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Minimal Workspace Bar */}
        <header className="h-16 bg-white border-b border-slate-200/90 px-6 flex items-center justify-between text-sm shrink-0 shadow-2xs">
          <div className="flex items-center gap-4">
            <span className="font-semibold text-slate-700 flex items-center gap-2">
              <span className="text-slate-400">Tổ chức:</span>
              <strong className="text-blue-700 bg-blue-50 px-2.5 py-1 rounded-lg border border-blue-200/80 font-mono text-xs">
                {user?.tenant_id || 'default_tenant'}
              </strong>
            </span>
            <span className="text-slate-200">|</span>
            <span className="text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-lg border border-emerald-200 text-xs font-semibold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Safe Multi-Agent System</span>
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowUpload(true)}
              className="px-4 py-2 text-xs font-bold rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-sm shadow-blue-500/20 transition-all duration-150 flex items-center gap-1.5 cursor-pointer active:scale-95"
            >
              <span>➕</span> Tải lên Hợp đồng
            </button>
            <button
              onClick={handleLogout}
              className="px-3 py-2 text-xs font-bold rounded-xl bg-slate-100 hover:bg-red-50 text-slate-600 hover:text-red-700 border border-slate-200 transition-all duration-150 flex items-center gap-1.5 cursor-pointer"
              title="Đăng xuất khỏi hệ thống"
            >
              <span>🚪</span> Đăng xuất
            </button>
          </div>
        </header>

        {/* Content Area - Persistent Tabs */}
        <main className="flex-1 flex overflow-hidden relative bg-slate-50">
          {/* Tab 1: QA */}
          <div className={`flex-1 flex flex-col h-full overflow-hidden ${activeTab === 'qa' ? '' : 'hidden'}`}>
            <ChatInterface
              documents={documents}
              token={token}
              apiUrl={API_URL}
              user={user}
              onLogout={handleLogout}
              onOpenUpload={() => setShowUpload(true)}
              resetSignal={chatResetSignal}
            />
          </div>

          {/* Tab 2: Compare */}
          <div className={`flex-1 flex flex-col h-full overflow-hidden ${activeTab === 'compare' ? '' : 'hidden'}`}>
            <ContractCompareView
              documents={documents}
              token={token}
              apiUrl={API_URL}
              onOpenUpload={() => setShowUpload(true)}
            />
          </div>

          {/* Tab 3: Risk */}
          <div className={`flex-1 flex flex-col h-full overflow-hidden ${activeTab === 'risk' ? '' : 'hidden'}`}>
            <RiskReviewDashboard
              documents={documents}
              token={token}
              apiUrl={API_URL}
              onOpenUpload={() => setShowUpload(true)}
            />
          </div>

          {/* Tab 4: Library */}
          <div className={`flex-1 flex flex-col h-full overflow-y-auto ${activeTab === 'library' ? '' : 'hidden'}`}>
            <div className="p-6 max-w-7xl mx-auto w-full space-y-6">
              {/* Header & Controls */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold text-slate-800">📚 Kho Hợp đồng & Văn bản Pháp lý</h2>
                    <span className="px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 text-xs font-bold border border-blue-200">
                      {documents.length} Hợp đồng
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Được phân quyền cho vai trò <strong className="text-slate-700">{user?.role}</strong> thuộc tổ chức{' '}
                    <strong className="text-slate-700">{user?.tenant_id}</strong>.
                  </p>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <input
                    type="text"
                    placeholder="Tìm kiếm theo tên hợp đồng, định dạng..."
                    value={docSearchQuery}
                    onChange={(e) => setDocSearchQuery(e.target.value)}
                    className="px-3.5 py-2 bg-slate-50 border border-slate-300 rounded-xl text-xs focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none w-full sm:w-72"
                  />
                  <button
                    onClick={() => setShowUpload(true)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-xl shadow-xs transition shrink-0 flex items-center gap-1.5"
                  >
                    <span>➕</span> Upload
                  </button>
                </div>
              </div>

              {/* Document Grid */}
              {filteredDocs.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredDocs.map((d) => (
                    <div
                      key={d.id}
                      className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs hover:shadow-md hover:border-blue-300 transition-all flex flex-col justify-between space-y-4 group"
                    >
                      <div>
                        <div className="flex items-start justify-between gap-2">
                          <div className="w-12 h-12 rounded-xl bg-white border border-slate-200/90 p-1.5 shadow-2xs flex items-center justify-center group-hover:scale-105 transition-transform duration-200 shrink-0">
                            <img src="/oulogo.png" alt="OU Document" className="w-full h-full object-contain" />
                          </div>
                          <span className="text-[11px] uppercase font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                            {d.file_type || 'PDF'}
                          </span>
                        </div>

                        <div className="mt-3">
                          <h3 className="font-bold text-sm text-slate-800 truncate" title={d.filename}>
                            {d.filename}
                          </h3>
                          <p className="text-[11px] text-slate-400 mt-1">
                            Tải lên: {d.created_at ? new Date(d.created_at).toLocaleDateString('vi-VN') : 'Gần đây'}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                          <span>{d.page_count || 1} trang</span>
                          <span className="text-emerald-600 font-semibold flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> ĐÃ SẴN SÀNG (READY)
                          </span>
                        </div>

                        {/* Action Shortcuts */}
                        <div className="grid grid-cols-3 gap-1.5 pt-1">
                          <button
                            onClick={() => setActiveTab('qa')}
                            className="px-2 py-1.5 bg-slate-50 hover:bg-blue-50 hover:text-blue-700 text-slate-600 font-semibold text-[11px] rounded-lg border border-slate-200 transition text-center cursor-pointer"
                          >
                            💬 Tra cứu
                          </button>
                          <button
                            onClick={() => setActiveTab('risk')}
                            className="px-2 py-1.5 bg-slate-50 hover:bg-amber-50 hover:text-amber-700 text-slate-600 font-semibold text-[11px] rounded-lg border border-slate-200 transition text-center cursor-pointer"
                          >
                            🛡️ Review
                          </button>
                          <button
                            onClick={() => setActiveTab('compare')}
                            className="px-2 py-1.5 bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 text-slate-600 font-semibold text-[11px] rounded-lg border border-slate-200 transition text-center cursor-pointer"
                          >
                            ⚖️ So sánh
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-16 bg-white rounded-2xl border border-dashed border-slate-300 p-8">
                  <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center text-3xl mx-auto mb-4">
                    📂
                  </div>
                  <h3 className="text-base font-bold text-slate-800">
                    {docSearchQuery ? 'Không tìm thấy hợp đồng phù hợp' : 'Kho Hợp đồng đang trống'}
                  </h3>
                  <p className="text-xs text-slate-500 max-w-md mx-auto mt-1 mb-6">
                    {docSearchQuery
                      ? 'Hãy thử tìm kiếm với từ khóa khác hoặc xóa bộ lọc.'
                      : 'Hãy tải lên các hợp đồng mẫu trong thư mục sample_contracts (.txt, .pdf, .docx) để bắt đầu hỏi đáp, quét rủi ro và so sánh.'}
                  </p>
                  {docSearchQuery ? (
                    <button
                      onClick={() => setDocSearchQuery('')}
                      className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs rounded-xl transition"
                    >
                      Xóa bộ lọc
                    </button>
                  ) : (
                    <button
                      onClick={() => setShowUpload(true)}
                      className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-xl shadow-sm transition cursor-pointer"
                    >
                      ➕ Tải lên Hợp đồng Đầu tiên
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <UploadModal
          token={token}
          apiUrl={API_URL}
          onClose={() => setShowUpload(false)}
          onUploadSuccess={() => {
            setShowUpload(false)
            fetchDocuments()
          }}
        />
      )}
    </div>
  )
}


