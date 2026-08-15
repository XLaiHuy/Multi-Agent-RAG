import { useState, useEffect } from 'react'
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

  const fetchDocuments = async () => {
    if (!token) return
    try {
      const res = await fetch(`${API_URL}/api/v1/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setDocuments(data)
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

  if (!token) {
    return <Login onLogin={handleLogin} apiUrl={API_URL} />
  }

  return (
    <div className="flex flex-col h-screen bg-slate-100 text-slate-900 font-sans antialiased overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between shadow-xs z-30 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-700 via-blue-600 to-indigo-500 text-white flex items-center justify-center font-black text-xl shadow-md">
            ⚖️
          </div>
          <div>
            <h1 className="font-extrabold text-base tracking-tight text-slate-800 flex items-center gap-2">
              Enterprise Contract Intelligence
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 font-bold border border-indigo-200 uppercase">
                Adaptive RAG v2
              </span>
            </h1>
            <div className="text-xs text-slate-500 font-medium flex items-center gap-2">
              <span>🏢 Tenant: <strong className="text-slate-700">{user?.tenant_id || 'default'}</strong></span>
            </div>
          </div>
        </div>

        {/* Center Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200">
          <button
            onClick={() => setActiveTab('qa')}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'qa'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <span>💬</span> Contract QA
          </button>
          <button
            onClick={() => setActiveTab('compare')}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'compare'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <span>⚖️</span> Compare
          </button>
          <button
            onClick={() => setActiveTab('risk')}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'risk'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <span>🛡️</span> Risk Review
          </button>
          <button
            onClick={() => setActiveTab('library')}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
              activeTab === 'library'
                ? 'bg-white text-indigo-700 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <span>📚</span> Library ({documents.length})
          </button>
        </nav>

        {/* User Info & Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowUpload(true)}
            className="px-3.5 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition flex items-center gap-1.5"
          >
            <span>➕</span> Upload
          </button>

          <div className="flex items-center gap-2 pl-2 border-l border-slate-200 text-xs">
            <span className="w-8 h-8 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center font-bold">
              {user?.username?.slice(0, 2).toUpperCase() || 'U'}
            </span>
            <div className="hidden lg:block text-left">
              <div className="font-bold text-slate-800 leading-none">{user?.full_name || user?.username}</div>
              <div className="text-[10px] text-slate-500 uppercase font-semibold mt-0.5">Role: {user?.role}</div>
            </div>
            <button
              onClick={handleLogout}
              className="text-slate-400 hover:text-red-600 text-xs font-medium p-1 transition"
              title="Logout"
            >
              🚪
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Body */}
      <main className="flex-1 flex overflow-hidden">
        {activeTab === 'qa' && (
          <ChatInterface documents={documents} token={token} apiUrl={API_URL} user={user} />
        )}
        {activeTab === 'compare' && (
          <ContractCompareView documents={documents} token={token} apiUrl={API_URL} />
        )}
        {activeTab === 'risk' && (
          <RiskReviewDashboard documents={documents} token={token} apiUrl={API_URL} />
        )}
        {activeTab === 'library' && (
          <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-slate-800">📚 Document Library</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Contracts accessible to your role (<strong className="text-slate-700">{user?.role}</strong>) under tenant{' '}
                  <strong className="text-slate-700">{user?.tenant_id}</strong>.
                </p>
              </div>
              <button
                onClick={() => setShowUpload(true)}
                className="px-4 py-2 bg-indigo-600 text-white font-bold text-xs rounded-xl shadow-sm hover:bg-indigo-700 transition"
              >
                Upload Contract
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {documents.map((d) => (
                <div key={d.id} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm hover:border-indigo-200 transition space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-2xl">📄</span>
                    <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                      {d.file_type}
                    </span>
                  </div>
                  <div>
                    <h3 className="font-bold text-sm text-slate-800 truncate" title={d.filename}>{d.filename}</h3>
                    <p className="text-xs text-slate-400 mt-0.5">Uploaded {new Date(d.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                    <span>{d.page_count || 1} pages</span>
                    <span className="text-emerald-600 font-semibold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> READY
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

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
