import { useState } from 'react'

export default function CitationViewerModal({ citation, index, onClose, onOpenWorkspace }) {
  const [copied, setCopied] = useState(false)
  if (!citation) return null

  const handleCopy = () => {
    navigator.clipboard.writeText(citation.supporting_text || '').catch(() => {})
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const docTitle = citation.filename || citation.document_id || 'Tài liệu Hợp đồng'
  const sectionPathStr = Array.isArray(citation.section_path) 
    ? citation.section_path.join(' > ') 
    : (citation.section_path || 'Phần chính')

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/70 backdrop-blur-sm p-4 animate-fadeIn"
      onClick={onClose}
    >
      <div 
        className="bg-white dark:bg-slate-900 rounded-2xl max-w-2xl w-full shadow-2xl border border-slate-200 dark:border-slate-800 overflow-hidden flex flex-col max-h-[85vh] animate-scaleUp"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 bg-gradient-to-r from-blue-900 via-indigo-900 to-blue-800 text-white flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white/15 flex items-center justify-center text-lg shadow-inner">
              📜
            </div>
            <div>
              <div className="font-bold text-base tracking-tight flex items-center gap-2">
                Căn cứ Trích dẫn #{index !== undefined ? index + 1 : ''}
                {citation.score && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-400/30">
                    Độ khớp: {Math.round(citation.score * 100)}%
                  </span>
                )}
              </div>
              <div className="text-xs text-blue-200/90 font-medium truncate max-w-md">
                {docTitle}
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-lg transition-colors cursor-pointer"
          >
            &times;
          </button>
        </div>

        {/* Location Metadata Strip */}
        <div className="px-6 py-2.5 bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between text-xs text-slate-600 dark:text-slate-400 gap-2">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 font-semibold text-slate-700 dark:text-slate-300">
              <span>📄</span> Trang {citation.page || 1}
            </span>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span className="flex items-center gap-1 text-blue-700 dark:text-blue-400 font-medium bg-blue-50 dark:bg-blue-950/40 px-2 py-0.5 rounded-md border border-blue-100 dark:border-blue-900">
              <span>📌</span> {sectionPathStr}
            </span>
          </div>

          {citation.bbox && (
            <span className="text-slate-500 dark:text-slate-400 font-mono text-[11px] bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-700">
              Toạ độ BBox: [{citation.bbox.x0}, {citation.bbox.y0}, {citation.bbox.x1}, {citation.bbox.y1}]
            </span>
          )}
        </div>

        {/* Verbatim Supporting Text Body */}
        <div className="p-6 overflow-y-auto flex-1 bg-white dark:bg-slate-900">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">
            Đoạn điều khoản nguyên văn làm căn cứ:
          </div>
          <div className="p-4 rounded-xl bg-amber-50/60 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-900/40 text-slate-800 dark:text-slate-200 text-sm leading-relaxed font-serif shadow-inner whitespace-pre-wrap selection:bg-amber-200">
            {citation.supporting_text}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-slate-50 dark:bg-slate-950 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="text-xs text-slate-500 dark:text-slate-400">
            Block ID: <code className="text-slate-700 dark:text-slate-300 bg-slate-200 dark:bg-slate-800 px-1 py-0.5 rounded">{citation.block_id || 'n/a'}</code>
          </div>
          <div className="flex gap-2">
            {onOpenWorkspace && (
              <button
                onClick={() => {
                  onClose()
                  onOpenWorkspace(citation)
                }}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 hover:bg-blue-100 transition shadow-sm active:scale-95 flex items-center gap-1.5 cursor-pointer"
              >
                🔍 Mở trong Bản đối soát PDF
              </button>
            )}
            <button
              onClick={handleCopy}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition shadow-sm active:scale-95 flex items-center gap-1.5 cursor-pointer"
            >
              {copied ? '✅ Đã sao chép!' : '📋 Sao chép đoạn văn'}
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition shadow-sm active:scale-95 cursor-pointer"
            >
              Đóng
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
