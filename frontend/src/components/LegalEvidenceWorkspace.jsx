import React, { useState, useEffect, useRef } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { Skeleton } from './ui/skeleton'
import { Alert, AlertDescription } from './ui/alert'
import { getAuthToken, API_BASE_URL } from '../api/client'

// Configure PDF.js worker
if (typeof window !== 'undefined' && 'Worker' in window) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).toString()
}

export default function LegalEvidenceWorkspace({
  documentId,
  documentTitle,
  fileType = 'pdf',
  targetPage = 1,
  targetBbox = null,
  supportingText = '',
  onClose,
}) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [pdfDoc, setPdfDoc] = useState(null)
  const [currentPage, setCurrentPage] = useState(targetPage || 1)
  const [numPages, setNumPages] = useState(1)
  const [scale, setScale] = useState(1.2)
  const [textContent, setTextContent] = useState('')

  const canvasRef = useRef(null)
  const renderTaskRef = useRef(null)

  // Sync targetPage when prop changes
  useEffect(() => {
    if (targetPage && targetPage > 0) {
      setCurrentPage(targetPage)
    }
  }, [targetPage])

  // Fetch document content
  useEffect(() => {
    if (!documentId) return

    let isMounted = true
    setLoading(true)
    setError(null)

    const token = getAuthToken()
    const contentUrl = `${API_BASE_URL}/documents/${documentId}/content`

    const isPdf = fileType.toLowerCase() === 'pdf' || (documentTitle && documentTitle.toLowerCase().endsWith('.pdf'))

    if (isPdf) {
      const loadingTask = pdfjsLib.getDocument({
        url: contentUrl,
        httpHeaders: token ? { Authorization: `Bearer ${token}` } : {},
      })

      loadingTask.promise
        .then((doc) => {
          if (!isMounted) return
          setPdfDoc(doc)
          setNumPages(doc.numPages)
          setLoading(false)
        })
        .catch((err) => {
          if (!isMounted) return
          setError(`Không thể tải tài liệu PDF (${err.message})`)
          setLoading(false)
        })
    } else {
      // Non-PDF text/markdown/docx fetch
      fetch(contentUrl, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          return res.text()
        })
        .then((text) => {
          if (!isMounted) return
          setTextContent(text)
          setLoading(false)
        })
        .catch((err) => {
          if (!isMounted) return
          setError(`Không thể tải nội dung văn bản (${err.message})`)
          setLoading(false)
        })
    }

    return () => {
      isMounted = false
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel()
      }
    }
  }, [documentId, fileType, documentTitle])

  // Render PDF page to canvas
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return

    let isCancelled = false
    const pageNum = Math.min(Math.max(1, currentPage), numPages || 1)

    pdfDoc.getPage(pageNum).then((page) => {
      if (isCancelled) return

      const canvas = canvasRef.current
      const context = canvas.getContext('2d')
      const viewport = page.getViewport({ scale })

      // Handle High-DPI screens
      const outputScale = window.devicePixelRatio || 1
      canvas.width = Math.floor(viewport.width * outputScale)
      canvas.height = Math.floor(viewport.height * outputScale)
      canvas.style.width = `${Math.floor(viewport.width)}px`
      canvas.style.height = `${Math.floor(viewport.height)}px`

      const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null

      if (renderTaskRef.current) {
        renderTaskRef.current.cancel()
      }

      const renderContext = {
        canvasContext: context,
        transform: transform,
        viewport: viewport,
      }

      const renderTask = page.render(renderContext)
      renderTaskRef.current = renderTask

      renderTask.promise
        .then(() => {
          // If bbox is present and on the current target page, draw highlight rectangle
          if (targetBbox && currentPage === targetPage) {
            context.save()
            if (outputScale !== 1) {
              context.scale(outputScale, outputScale)
            }
            context.fillStyle = 'rgba(234, 179, 8, 0.25)' // Amber highlight
            context.strokeStyle = 'rgba(202, 138, 4, 0.9)'
            context.lineWidth = 2

            const { x0, y0, x1, y1 } = targetBbox
            const bw = (x1 - x0) * scale
            const bh = (y1 - y0) * scale
            const bx = x0 * scale
            const by = y0 * scale

            context.fillRect(bx, by, bw, bh)
            context.strokeRect(bx, by, bw, bh)
            context.restore()
          }
        })
        .catch(() => {
          // Render cancelled or page switch
        })
    })

    return () => {
      isCancelled = true
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel()
      }
    }
  }, [pdfDoc, currentPage, scale, numPages, targetBbox, targetPage])

  const isPdf = fileType.toLowerCase() === 'pdf' || (documentTitle && documentTitle.toLowerCase().endsWith('.pdf'))

  return (
    <Card className="flex flex-col h-full border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header & Controls */}
      <CardHeader className="p-3.5 border-b border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 backdrop-blur-sm">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2 min-w-0">
            <span className="text-lg">📜</span>
            <div className="min-w-0">
              <CardTitle className="text-sm font-semibold truncate text-slate-900 dark:text-white">
                {documentTitle || documentId || 'Không gian Đối soát Căn cứ Pháp lý'}
              </CardTitle>
              <div className="flex items-center gap-1.5 mt-0.5">
                <Badge variant="outline" className="text-[10px] uppercase font-mono px-1.5 py-0">
                  {fileType}
                </Badge>
                {isPdf && (
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    Trang {currentPage} / {numPages}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1">
            {isPdf && (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  onClick={() => setScale((s) => Math.max(0.6, s - 0.2))}
                  title="Thu nhỏ"
                >
                  🔍-
                </Button>
                <span className="text-xs font-mono text-slate-500 w-10 text-center">
                  {Math.round(scale * 100)}%
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  onClick={() => setScale((s) => Math.min(2.5, s + 0.2))}
                  title="Phóng to"
                >
                  🔍+
                </Button>
                <div className="w-[1px] h-4 bg-slate-200 dark:bg-slate-700 mx-1" />
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  disabled={currentPage <= 1}
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                >
                  ◀
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  disabled={currentPage >= numPages}
                  onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
                >
                  ▶
                </Button>
              </>
            )}
            {onClose && (
              <Button
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0 ml-1 text-slate-500 hover:text-slate-900 dark:hover:text-white"
                onClick={onClose}
                title="Đóng bảng đối soát"
              >
                ✕
              </Button>
            )}
          </div>
        </div>

        {/* Supporting snippet banner if active */}
        {supportingText && (
          <div className="mt-2 text-xs p-2 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 text-amber-900 dark:text-amber-300">
            <span className="font-semibold">Đoạn trích dẫn mục tiêu (Trang {targetPage}):</span>
            <p className="mt-0.5 line-clamp-2 italic text-slate-700 dark:text-slate-300">
              "{supportingText}"
            </p>
          </div>
        )}
      </CardHeader>

      {/* Main Content Area */}
      <CardContent className="flex-1 p-4 overflow-auto flex items-center justify-center bg-slate-100/70 dark:bg-slate-950/90">
        {loading && (
          <div className="flex flex-col items-center justify-center space-y-3 p-8">
            <Skeleton className="w-72 h-96 rounded-lg" />
            <p className="text-xs text-slate-500">Đang tải và dựng văn bản gốc...</p>
          </div>
        )}

        {error && (
          <Alert variant="destructive" className="max-w-md m-auto">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {!loading && !error && isPdf && (
          <div className="relative shadow-lg border border-slate-300 dark:border-slate-700 bg-white rounded">
            <canvas ref={canvasRef} className="block max-w-full" />
          </div>
        )}

        {!loading && !error && !isPdf && (
          <div className="w-full h-full p-4 overflow-auto rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 font-mono text-xs leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
            {textContent || 'Tài liệu không có nội dung văn bản.'}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
