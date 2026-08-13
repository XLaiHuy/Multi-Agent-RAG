import { useState } from 'react'

export default function UploadModal({ token, onClose }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle') // idle, uploading, success, error
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const BLUE = '#1d4ed8'
  const BLUE_LIGHT = '#eff6ff'

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setStatus('idle')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setStatus('uploading')
    setError('')
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload thất bại')
      
      setResult(data)
      setStatus('success')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        background: 'white', width: '100%', maxWidth: '500px',
        borderRadius: '20px', padding: '32px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.2)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h2 style={{ margin: 0, fontSize: '20px', fontWeight: '800', color: '#0f172a' }}>
            Tải lên tài liệu & Hình ảnh
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', fontSize: '24px',
            color: '#64748b', cursor: 'pointer', padding: 0
          }}>&times;</button>
        </div>

        <div style={{
          border: `2px dashed ${file ? BLUE : '#cbd5e1'}`,
          borderRadius: '16px', padding: '40px 20px',
          textAlign: 'center', background: file ? BLUE_LIGHT : '#f8fafc',
          marginBottom: '24px', transition: 'all 0.2s'
        }}>
          <input
            type="file"
            id="file-upload"
            accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <div style={{ fontSize: '40px', marginBottom: '12px' }}>📁</div>
            {file ? (
              <div>
                <div style={{ fontWeight: '600', color: BLUE, fontSize: '15px', marginBottom: '4px' }}>{file.name}</div>
                <div style={{ color: '#64748b', fontSize: '12px' }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
              </div>
            ) : (
              <div>
                <div style={{ fontWeight: '600', color: '#334155', fontSize: '15px', marginBottom: '4px' }}>
                  Bấm để chọn file PDF, TXT, MD hoặc Ảnh (PNG, JPG, WEBP)
                </div>
                <div style={{ color: '#94a3b8', fontSize: '12px' }}>
                  Hỗ trợ Multimodal Vision OCR bóc tách bảng số liệu & sơ đồ
                </div>
              </div>
            )}
          </label>
        </div>

        {status === 'error' && (
          <div style={{ padding: '12px', background: '#fef2f2', color: '#dc2626', borderRadius: '8px', fontSize: '13px', marginBottom: '20px' }}>
            ⚠️ {error}
          </div>
        )}

        {status === 'success' && result && (
          <div style={{ padding: '16px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '12px', marginBottom: '20px' }}>
            <div style={{ color: '#16a34a', fontWeight: '700', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>✅</span> Tải lên và Index thành công!
            </div>
            <ul style={{ margin: 0, paddingLeft: '20px', color: '#15803d', fontSize: '13px', lineHeight: 1.6 }}>
              <li>File: {result.filename}</li>
              <li>Tạo thành <strong>{result.chunks_generated}</strong> chunks</li>
              <li>Đã lưu vào Vector Database</li>
            </ul>
          </div>
        )}

        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={{
            padding: '10px 20px', borderRadius: '10px', border: '1px solid #cbd5e1',
            background: 'white', color: '#475569', fontWeight: '600', cursor: 'pointer',
            fontSize: '14px', fontFamily: 'Inter, sans-serif'
          }}>
            Đóng
          </button>
          
          <button
            onClick={handleUpload}
            disabled={!file || status === 'uploading' || status === 'success'}
            style={{
              padding: '10px 24px', borderRadius: '10px', border: 'none',
              background: (!file || status === 'uploading' || status === 'success') ? '#94a3b8' : BLUE,
              color: 'white', fontWeight: '600', cursor: (!file || status === 'uploading' || status === 'success') ? 'not-allowed' : 'pointer',
              fontSize: '14px', fontFamily: 'Inter, sans-serif',
              boxShadow: (!file || status === 'uploading' || status === 'success') ? 'none' : '0 4px 12px rgba(29,78,216,0.3)',
            }}
          >
            {status === 'uploading' ? 'Đang xử lý...' : 'Tải lên & Xử lý'}
          </button>
        </div>
      </div>
    </div>
  )
}
