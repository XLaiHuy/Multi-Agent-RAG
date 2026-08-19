import React, { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import ContractCompareView from './components/ContractCompareView'
import RiskReviewDashboard from './components/RiskReviewDashboard'
import LegalEvidenceWorkspace from './components/LegalEvidenceWorkspace'
import UploadModal from './components/UploadModal'
import Login from './components/Login'
import { useDocuments } from './hooks/useDocuments'
import { setAuthToken } from './api/client'
import { Button } from './components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card'
import { Badge } from './components/ui/badge'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [token, setTokenState] = useState(localStorage.getItem('rag_token') || localStorage.getItem('token') || '')
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('user') || 'null')
    } catch {
      return null
    }
  })
  const [activeTab, setActiveTab] = useState('qa') // 'qa' | 'compare' | 'risk' | 'library'
  const [showUpload, setShowUpload] = useState(false)
  const [docSearchQuery, setDocSearchQuery] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [chatResetSignal, setChatResetSignal] = useState(0)

  // Split-pane Evidence Workspace state
  const [workspaceDoc, setWorkspaceDoc] = useState(null)

  // TanStack Query Documents
  const { data: documents = [], isLoading: _docsLoading, refetch: _refetchDocs } = useDocuments()

  useEffect(() => {
    const handleUnauthorized = () => {
      handleLogout()
    }
    window.addEventListener('auth:unauthorized', handleUnauthorized)
    return () => window.removeEventListener('auth:unauthorized', handleUnauthorized)
  }, [])

  const handleLogin = (accessToken, userInfo) => {
    setTokenState(accessToken)
    setUser(userInfo)
    setAuthToken(accessToken)
    localStorage.setItem('token', accessToken)
    localStorage.setItem('user', JSON.stringify(userInfo))
  }

  const handleLogout = () => {
    setTokenState('')
    setUser(null)
    setAuthToken('')
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setWorkspaceDoc(null)
  }

  const handleNewChat = () => {
    setActiveTab('qa')
    setChatResetSignal((prev) => prev + 1)
  }

  const handleOpenWorkspace = (citationOrDoc) => {
    if (!citationOrDoc) return
    const docId = citationOrDoc.document_id || citationOrDoc.id
    const filename = citationOrDoc.filename || citationOrDoc.original_filename || docId
    const ext = filename.split('.').pop() || citationOrDoc.file_type || 'pdf'

    setWorkspaceDoc({
      documentId: docId,
      documentTitle: filename,
      fileType: ext,
      targetPage: citationOrDoc.page || 1,
      targetBbox: citationOrDoc.bbox || null,
      supportingText: citationOrDoc.supporting_text || '',
    })
  }

  if (!token) {
    return <Login onLogin={handleLogin} apiUrl={API_URL} />
  }

  const filteredDocs = documents.filter(
    (d) =>
      (d.filename || '').toLowerCase().includes(docSearchQuery.toLowerCase()) ||
      (d.file_type || '').toLowerCase().includes(docSearchQuery.toLowerCase())
  )

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans antialiased overflow-hidden selection:bg-blue-500 selection:text-white">
      {/* 1. LEFT SIDEBAR */}
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

            {/* Workspace Active Indicator */}
            {workspaceDoc && sidebarOpen && (
              <div className="p-3 bg-blue-50/70 dark:bg-blue-950/30 rounded-xl border border-blue-200/80 dark:border-blue-900/50">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-bold text-blue-800 dark:text-blue-300">
                    📜 Bản đối soát PDF
                  </span>
                  <button
                    onClick={() => setWorkspaceDoc(null)}
                    className="text-xs text-blue-600 hover:text-blue-900 font-bold cursor-pointer"
                  >
                    ✕ Đóng
                  </button>
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 truncate">
                  {workspaceDoc.documentTitle}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Bottom User Info & Prominent Logout */}
        <div className="p-3 border-t border-slate-100 bg-slate-50/60 space-y-2">
          {/* User Profile Card */}
          <div className="flex items-center justify-between gap-2.5 p-2.5 rounded-xl bg-white border border-slate-200/90 shadow-2xs">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center font-bold text-xs shadow-xs shrink-0 ring-2 ring-blue-100">
                {user?.username?.slice(0, 2).toUpperCase() || 'AD'}
              </div>
              {sidebarOpen && (
                <div className="flex flex-col truncate">
                  <span className="text-xs font-bold text-slate-800 truncate leading-tight">
                    {user?.full_name || user?.username || 'Admin'}
                  </span>
                  <span className="text-[11px] text-slate-500 truncate mt-0.5">
                    {user?.role?.toUpperCase() || 'LEGAL'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Prominent Red Logout Button */}
          <button
            onClick={handleLogout}
            className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-bold text-red-600 hover:text-white bg-red-50 hover:bg-red-600 border border-red-200 hover:border-red-600 transition-all duration-200 shadow-2xs active:scale-95 cursor-pointer`}
            title="Đăng xuất khỏi hệ thống"
          >
            <span>🚪</span>
            {sidebarOpen && <span>Đăng xuất</span>}
          </button>
        </div>
      </aside>

      {/* 2. MAIN APPLICATION CONTENT (With Split Workspace support) */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header Bar */}
        <header className="h-16 bg-white/95 backdrop-blur-sm border-b border-slate-200/90 px-6 flex items-center justify-between shrink-0 z-30 shadow-2xs">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-lg">
                {activeTab === 'qa' && '💬'}
                {activeTab === 'risk' && '🛡️'}
                {activeTab === 'compare' && '⚖️'}
                {activeTab === 'library' && '📚'}
              </span>
              <h1 className="text-base font-extrabold text-slate-900 tracking-tight">
                {activeTab === 'qa' && 'Trợ lý AI Tra cứu & Hỏi đáp Pháp lý'}
                {activeTab === 'risk' && 'Thẩm định & Quét Rủi ro Hợp đồng'}
                {activeTab === 'compare' && 'So sánh & Đối chiếu Điều khoản Hợp đồng'}
                {activeTab === 'library' && 'Thư viện & Kho Hợp đồng Doanh nghiệp'}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              onClick={() => setShowUpload(true)}
              size="sm"
              className="font-bold flex items-center gap-1.5 shadow-sm shadow-blue-500/20"
            >
              <span>+ Tải hợp đồng</span>
            </Button>
            <Button
              onClick={handleLogout}
              variant="outline"
              size="sm"
              className="text-red-600 border-red-200 hover:bg-red-50"
            >
              🚪 Thoát
            </Button>
          </div>
        </header>

        {/* Tab Views Container (Split layout when workspaceDoc active) */}
        <div className="flex-1 flex h-[calc(100vh-64px)] overflow-hidden">
          {/* Main Active Tab Left Pane */}
          <div
            className={`flex-1 flex flex-col h-full overflow-hidden transition-all duration-300 ${
              workspaceDoc ? 'w-1/2 border-r border-slate-200/90' : 'w-full'
            }`}
          >
            {activeTab === 'qa' && (
              <ChatInterface
                documents={documents}
                token={token}
                apiUrl={API_URL}
                user={user}
                onLogout={handleLogout}
                onOpenUpload={() => setShowUpload(true)}
                onOpenWorkspace={handleOpenWorkspace}
                resetSignal={chatResetSignal}
              />
            )}

            {activeTab === 'risk' && (
              <RiskReviewDashboard
                documents={documents}
                token={token}
                apiUrl={API_URL}
                onOpenUpload={() => setShowUpload(true)}
                onOpenWorkspace={handleOpenWorkspace}
              />
            )}

            {activeTab === 'compare' && (
              <ContractCompareView
                documents={documents}
                token={token}
                apiUrl={API_URL}
                onOpenUpload={() => setShowUpload(true)}
                onOpenWorkspace={handleOpenWorkspace}
              />
            )}

            {activeTab === 'library' && (
              <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
                <Card>
                  <CardHeader className="p-6 pb-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div>
                        <CardTitle className="text-lg font-bold">Kho Hợp đồng Doanh nghiệp</CardTitle>
                        <p className="text-xs text-slate-500 mt-1">
                          Tổng số {documents.length} văn bản đã được phân tách và lập chỉ mục tìm kiếm lai.
                        </p>
                      </div>
                      <div className="w-full sm:w-72">
                        <input
                          type="text"
                          value={docSearchQuery}
                          onChange={(e) => setDocSearchQuery(e.target.value)}
                          placeholder="🔍 Tìm kiếm theo tên hoặc định dạng..."
                          className="w-full px-3.5 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="p-6 pt-0">
                    {filteredDocs.length === 0 ? (
                      <div className="p-12 text-center text-slate-400 space-y-3">
                        <div className="text-4xl">📂</div>
                        <p className="text-sm font-medium">Chưa có hợp đồng nào phù hợp với tìm kiếm.</p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredDocs.map((doc) => (
                          <Card
                            key={doc.id}
                            className="p-4 hover:border-blue-300 hover:shadow-md transition-all flex flex-col justify-between"
                          >
                            <div>
                              <div className="flex items-center justify-between mb-2">
                                <Badge variant="outline" className="uppercase text-[10px]">
                                  {doc.file_type}
                                </Badge>
                                <span className="text-[11px] text-slate-400">
                                  {doc.created_at?.split('T')[0] || 'Vừa xong'}
                                </span>
                              </div>
                              <h4 className="font-bold text-sm text-slate-900 line-clamp-2 mb-1">
                                {doc.filename}
                              </h4>
                              <p className="text-xs text-slate-500">
                                {doc.char_count?.toLocaleString() || 0} ký tự • {doc.page_count || 1} trang
                              </p>
                            </div>

                            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                className="w-full text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                                onClick={() => handleOpenWorkspace(doc)}
                              >
                                🔍 Xem trong PDF
                              </Button>
                            </div>
                          </Card>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Right Pane: Split-view Legal Evidence Workspace */}
          {workspaceDoc && (
            <div className="w-1/2 flex flex-col h-full bg-slate-100/50 p-2 animate-in slide-in-from-right duration-300">
              <LegalEvidenceWorkspace
                documentId={workspaceDoc.documentId}
                documentTitle={workspaceDoc.documentTitle}
                fileType={workspaceDoc.fileType}
                targetPage={workspaceDoc.targetPage}
                targetBbox={workspaceDoc.targetBbox}
                supportingText={workspaceDoc.supportingText}
                onClose={() => setWorkspaceDoc(null)}
              />
            </div>
          )}
        </div>
      </main>

      {/* Upload Modal */}
      {showUpload && <UploadModal open={showUpload} onClose={() => setShowUpload(false)} />}
    </div>
  )
}
