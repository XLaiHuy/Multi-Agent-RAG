import { useState } from 'react'
import Login from './components/Login'
import ChatInterface from './components/ChatInterface'

function App() {
  const [token, setToken] = useState(localStorage.getItem('rag_token') || null)
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('rag_user')) || null)

  const handleLogin = (data) => {
    setToken(data.access_token)
    setUser(data.user_info)
    localStorage.setItem('rag_token', data.access_token)
    localStorage.setItem('rag_user', JSON.stringify(data.user_info))
  }

  const handleLogout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem('rag_token')
    localStorage.removeItem('rag_user')
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">
      {!token ? (
        <Login onLogin={handleLogin} />
      ) : (
        <ChatInterface token={token} user={user} onLogout={handleLogout} />
      )}
    </div>
  )
}

export default App
