import React, { useState, useEffect } from 'react'
import CitationViewerModal from './CitationViewerModal'
import { Button } from './ui/button'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { Alert, AlertDescription } from './ui/alert'
import { Skeleton } from './ui/skeleton'

const DEFAULT_FACETS = [
  'Thời hạn hợp đồng & Quyền chấm dứt (Term & Termination)',
  'Quy định thời hạn báo trước (Notice Period Requirements)',
  'Giới hạn trách nhiệm & Mức trần bồi thường (Limitation of Liability)',
  'Nghĩa vụ bồi thường bên thứ ba (Indemnification)',
  'Luật áp dụng & Cơ quan giải quyết tranh chấp (Governing Law & Forum)',
  'Điều khoản thanh toán & Điều chỉnh giá (Payment & Pricing)',
  'Bảo mật thông tin & Dữ liệu (Confidentiality & Data Protection)',
]

export default function ContractCompareView({
  documents = [],
  token,
  apiUrl,
  onOpenUpload: _onOpenUpload,
  onOpenWorkspace,
}) {
  const [contractA, setContractA] = useState(documents[0]?.id || '')
  const [contractB, setContractB] = useState(documents[1]?.id || documents[0]?.id || '')
  const [selectedFacets, setSelectedFacets] = useState(DEFAULT_FACETS)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [activeCitation, setActiveCitation] = useState(null)

  useEffect(() => {
    if (documents.length > 0) {
      if (!contractA) setContractA(documents[0].id)
      if (!contractB) setContractB(documents[1]?.id || documents[0].id)
    }
  }, [documents, contractA, contractB])

  const toggleFacet = (facet) => {
    setSelectedFacets((prev) =>
      prev.includes(facet) ? prev.filter((f) => f !== facet) : [...prev, facet]
    )
  }

  const handleCompare = async () => {
    if (!contractA || !contractB) {
      setError('Vui lòng chọn đủ 2 hợp đồng để tiến hành so sánh đối chiếu.')
      return
    }

    if (contractA === contractB) {
      setError('Vui lòng chọn 2 phiên bản hoặc 2 hợp đồng khác nhau để so sánh.')
      return
    }

    if (selectedFacets.length === 0) {
      setError('Vui lòng chọn ít nhất một khía cạnh điều khoản so sánh.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const res = await fetch(`${apiUrl}/api/v1/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          doc_a_id: contractA,
          doc_b_id: contractB,
          facets: selectedFacets,
        }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Lỗi khi so sánh đối chiếu hợp đồng.')
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const docAObj = documents.find((d) => d.id === contractA)
  const docBObj = documents.find((d) => d.id === contractB)

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-0 shadow-md">
        <CardHeader className="p-6">
          <CardTitle className="text-xl font-bold flex items-center gap-2 text-white">
            <span>⚖️</span> So sánh & Đối chiếu Điều khoản Hợp đồng (Contract Comparison)
          </CardTitle>
          <p className="text-slate-300 text-xs mt-1">
            Phân tích điểm khác biệt, rủi ro phát sinh giữa 2 bản hợp đồng hoặc 2 phiên bản cập nhật.
          </p>
        </CardHeader>
      </Card>

      {/* Contract Selectors & Facet Filters */}
      <Card>
        <CardContent className="p-6 space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Contract A */}
            <div>
              <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
                Hợp đồng A (Bản gốc / Chuẩn)
              </label>
              <select
                value={contractA}
                onChange={(e) => setContractA(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none text-slate-900 dark:text-white"
              >
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    📄 {d.filename} ({d.file_type.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            {/* Contract B */}
            <div>
              <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
                Hợp đồng B (Bản đối tác / Sửa đổi)
              </label>
              <select
                value={contractB}
                onChange={(e) => setContractB(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:outline-none text-slate-900 dark:text-white"
              >
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    📄 {d.filename} ({d.file_type.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Facets Multi-select */}
          <div>
            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
              Khía cạnh pháp lý cần so sánh
            </label>
            <div className="flex flex-wrap gap-2">
              {DEFAULT_FACETS.map((facet) => {
                const isSelected = selectedFacets.includes(facet)
                return (
                  <button
                    key={facet}
                    type="button"
                    onClick={() => toggleFacet(facet)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer ${
                      isSelected
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {isSelected ? '✓ ' : '+ '}
                    {facet}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <Button onClick={handleCompare} disabled={loading || !contractA || !contractB} className="h-11 px-8">
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="animate-spin">⏳</span> Đang so sánh điều khoản...
                </span>
              ) : (
                '⚖️ Bắt đầu Đối chiếu So sánh'
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full rounded-xl" />
          <Skeleton className="h-64 w-full rounded-xl" />
        </div>
      )}

      {/* Comparison Results */}
      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Executive Summary */}
          <Card className="border-indigo-200 dark:border-indigo-900/50 bg-indigo-50/40 dark:bg-indigo-950/20">
            <CardHeader className="p-5">
              <CardTitle className="text-base font-bold text-indigo-950 dark:text-indigo-200">
                Tổng hợp Đối chiếu Khác biệt
              </CardTitle>
            </CardHeader>
            <CardContent className="p-5 pt-0">
              <p className="text-xs leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
                {result.summary || 'Đã hoàn tất đối chiếu các điều khoản chính giữa hai tài liệu.'}
              </p>
            </CardContent>
          </Card>

          {/* Matrix of Facets */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
              Chi tiết Đối chiếu từng Khía cạnh ({result.facets?.length || 0})
            </h3>

            <div className="grid grid-cols-1 gap-4">
              {(result.facets || []).map((f, idx) => (
                <Card key={idx} className="overflow-hidden">
                  <CardHeader className="p-4 bg-slate-50/80 dark:bg-slate-900/80 border-b border-slate-100 dark:border-slate-800">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
                        {f.facet_name}
                      </CardTitle>
                      {f.has_conflict && (
                        <Badge variant="destructive" className="text-[10px] uppercase">
                          Có khác biệt lớn
                        </Badge>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="p-4 space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Doc A Clause */}
                      <div className="p-3 rounded-lg bg-blue-50/60 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/40 text-xs">
                        <div className="font-semibold text-blue-900 dark:text-blue-300 mb-1 flex items-center justify-between">
                          <span>{docAObj?.filename || 'Hợp đồng A'}:</span>
                          {onOpenWorkspace && f.citation_a && (
                            <button
                              onClick={() =>
                                onOpenWorkspace({
                                  document_id: contractA,
                                  filename: docAObj?.filename,
                                  ...f.citation_a,
                                })
                              }
                              className="text-[10px] text-blue-600 hover:underline cursor-pointer"
                            >
                              Xem PDF
                            </button>
                          )}
                        </div>
                        <p className="text-slate-700 dark:text-slate-300 leading-relaxed font-serif italic">
                          "{f.clause_a || 'Không tìm thấy điều khoản tương ứng.'}"
                        </p>
                      </div>

                      {/* Doc B Clause */}
                      <div className="p-3 rounded-lg bg-indigo-50/60 dark:bg-indigo-950/20 border border-indigo-200 dark:border-indigo-900/40 text-xs">
                        <div className="font-semibold text-indigo-900 dark:text-indigo-300 mb-1 flex items-center justify-between">
                          <span>{docBObj?.filename || 'Hợp đồng B'}:</span>
                          {onOpenWorkspace && f.citation_b && (
                            <button
                              onClick={() =>
                                onOpenWorkspace({
                                  document_id: contractB,
                                  filename: docBObj?.filename,
                                  ...f.citation_b,
                                })
                              }
                              className="text-[10px] text-indigo-600 hover:underline cursor-pointer"
                            >
                              Xem PDF
                            </button>
                          )}
                        </div>
                        <p className="text-slate-700 dark:text-slate-300 leading-relaxed font-serif italic">
                          "{f.clause_b || 'Không tìm thấy điều khoản tương ứng.'}"
                        </p>
                      </div>
                    </div>

                    {/* Synthesis Analysis */}
                    {f.analysis && (
                      <div className="pt-2 text-xs text-slate-700 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/60 p-3 rounded-lg">
                        <span className="font-semibold text-slate-900 dark:text-white">Nhận xét pháp lý: </span>
                        <span>{f.analysis}</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
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
