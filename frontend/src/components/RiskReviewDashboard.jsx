import React, { useState, useEffect } from 'react'
import CitationViewerModal from './CitationViewerModal'
import { Button } from './ui/button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Alert, AlertDescription } from './ui/alert'
import { Skeleton } from './ui/skeleton'

const SEVERITY_BADGES = {
  critical: 'destructive',
  high: 'warning',
  medium: 'warning',
  low: 'default',
}

const SEVERITY_ICONS = {
  critical: '🚨',
  high: '⚠️',
  medium: '⚡',
  low: 'ℹ️',
}

export default function RiskReviewDashboard({
  documents = [],
  token,
  apiUrl,
  onOpenUpload,
  onOpenWorkspace,
}) {
  const [selectedDoc, setSelectedDoc] = useState(documents[0]?.id || '')
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')
  const [activeCitation, setActiveCitation] = useState(null)

  useEffect(() => {
    if (!selectedDoc && documents.length > 0) {
      setSelectedDoc(documents[0].id)
    }
  }, [documents, selectedDoc])

  const handleReview = async () => {
    if (!selectedDoc) {
      setError('Vui lòng chọn tài liệu hợp đồng để quét rủi ro.')
      return
    }

    setLoading(true)
    setError('')
    setReport(null)

    try {
      const res = await fetch(`${apiUrl}/api/v1/risk/review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          document_id: selectedDoc,
        }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Lỗi khi phân tích rủi ro hợp đồng.')
      }

      const data = await res.json()
      setReport(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const selectedDocObj = documents.find((d) => d.id === selectedDoc)

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
      {/* Header Banner */}
      <Card className="bg-gradient-to-r from-slate-900 via-blue-950 to-slate-900 text-white border-0 shadow-md">
        <CardHeader className="p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <CardTitle className="text-xl font-bold flex items-center gap-2 text-white">
                <span>🛡️</span> Thẩm định & Quét Rủi ro Hợp đồng (Risk Review)
              </CardTitle>
              <CardDescription className="text-slate-300 text-xs mt-1">
                Kết hợp quy tắc vị từ định lượng xác định (Deterministic Predicates) và LLM Critic phân tích ngữ cảnh.
              </CardDescription>
            </div>
            {onOpenUpload && (
              <Button
                variant="secondary"
                size="sm"
                className="bg-white/10 hover:bg-white/20 text-white border-0"
                onClick={onOpenUpload}
              >
                + Tải hợp đồng mới
              </Button>
            )}
          </div>
        </CardHeader>
      </Card>

      {/* Contract Selector Controls */}
      <Card>
        <CardContent className="p-6 flex flex-col md:flex-row items-end gap-4">
          <div className="flex-1 w-full">
            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
              Chọn Hợp đồng trong Thư viện để Thẩm định
            </label>
            <select
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none text-slate-900 dark:text-white"
            >
              {documents.map((d) => (
                <option key={d.id} value={d.id}>
                  📄 {d.filename} ({d.file_type.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <Button
            onClick={handleReview}
            disabled={loading || !selectedDoc}
            className="w-full md:w-auto h-11 px-6 font-semibold"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">⏳</span> Đang thẩm định rủi ro...
              </span>
            ) : (
              '🔍 Bắt đầu Quét Rủi ro'
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full rounded-xl" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Skeleton className="h-48 rounded-xl" />
            <Skeleton className="h-48 rounded-xl" />
          </div>
        </div>
      )}

      {/* Risk Report View */}
      {report && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Executive Summary Card */}
          <Card className="border-blue-200 dark:border-blue-900/50 bg-blue-50/40 dark:bg-blue-950/20">
            <CardHeader className="p-5">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base font-bold text-blue-900 dark:text-blue-200">
                    Tổng kết Đánh giá Mức độ Rủi ro
                  </CardTitle>
                  <p className="text-xs text-blue-700 dark:text-blue-400 mt-0.5">
                    Hợp đồng: <strong>{selectedDocObj?.filename || selectedDoc}</strong>
                  </p>
                </div>
                <Badge
                  variant={
                    report.overall_risk === 'high' || report.overall_risk === 'critical'
                      ? 'destructive'
                      : report.overall_risk === 'medium'
                      ? 'warning'
                      : 'success'
                  }
                  className="text-xs px-3 py-1 uppercase tracking-wider font-bold"
                >
                  Mức độ: {report.overall_risk || 'Chưa phân loại'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-5 pt-0">
              <p className="text-xs leading-relaxed text-slate-700 dark:text-slate-300">
                {report.summary || 'Đã hoàn tất rà soát toàn bộ các điều khoản theo bộ quy tắc pháp luật Việt Nam và chuẩn mực quốc tế.'}
              </p>
            </CardContent>
          </Card>

          {/* Identified Risk Items */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400 flex items-center gap-2">
              <span>⚠️</span> Danh sách Điều khoản Rủi ro ({report.risks?.length || 0})
            </h3>

            {(!report.risks || report.risks.length === 0) ? (
              <Card>
                <CardContent className="p-8 text-center text-slate-500">
                  ✅ Không phát hiện điều khoản rủi ro vi phạm pháp luật nghiêm trọng nào trong hợp đồng này.
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {report.risks.map((risk, idx) => (
                  <Card key={idx} className="overflow-hidden hover:shadow-md transition-shadow">
                    <CardHeader className="p-4 bg-slate-50/80 dark:bg-slate-900/80 border-b border-slate-100 dark:border-slate-800">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-base shrink-0">
                            {SEVERITY_ICONS[risk.severity] || '⚠️'}
                          </span>
                          <CardTitle className="text-sm font-bold text-slate-900 dark:text-white truncate">
                            {risk.rule_name || risk.risk_type}
                          </CardTitle>
                        </div>
                        <Badge variant={SEVERITY_BADGES[risk.severity] || 'default'} className="uppercase text-[10px]">
                          {risk.severity}
                        </Badge>
                      </div>
                    </CardHeader>

                    <CardContent className="p-4 space-y-3 text-xs">
                      <div>
                        <span className="font-semibold text-slate-700 dark:text-slate-300">Đánh giá rủi ro: </span>
                        <span className="text-slate-600 dark:text-slate-400">{risk.explanation || risk.description}</span>
                      </div>

                      {risk.clause_text && (
                        <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 font-serif text-slate-800 dark:text-slate-200 text-xs leading-relaxed italic">
                          "{risk.clause_text}"
                        </div>
                      )}

                      {risk.recommendation && (
                        <div className="p-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/40 text-emerald-800 dark:text-emerald-300">
                          <span className="font-semibold">Đề xuất chỉnh sửa: </span>
                          <span>{risk.recommendation}</span>
                        </div>
                      )}

                      {risk.citation && (
                        <div className="pt-1 flex items-center justify-between">
                          <span className="text-[11px] text-slate-500">
                            Trang {risk.citation.page || 1} • {risk.citation.section_path?.join(' > ') || 'Điều khoản'}
                          </span>
                          <div className="flex gap-2">
                            {onOpenWorkspace && (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-7 text-xs text-blue-600 hover:text-blue-700"
                                onClick={() =>
                                  onOpenWorkspace({
                                    document_id: selectedDoc,
                                    filename: selectedDocObj?.filename,
                                    ...risk.citation,
                                  })
                                }
                              >
                                🔍 Xem trong PDF
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 text-xs"
                              onClick={() => setActiveCitation(risk.citation)}
                            >
                              Chi tiết trích dẫn
                            </Button>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Citation Modal */}
      {activeCitation && (
        <CitationViewerModal
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
          onOpenWorkspace={onOpenWorkspace}
        />
      )}
    </div>
  )
}
