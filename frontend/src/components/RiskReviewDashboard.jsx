import { useState, useEffect } from 'react'
import CitationViewerModal from './CitationViewerModal'

const SEVERITY_COLORS = {
  critical: 'bg-red-500/10 text-red-700 border-red-300 ring-red-500/20',
  high: 'bg-orange-500/10 text-orange-700 border-orange-300 ring-orange-500/20',
  medium: 'bg-amber-500/10 text-amber-700 border-amber-300 ring-amber-500/20',
  low: 'bg-blue-500/10 text-blue-700 border-blue-300 ring-blue-500/20',
}

const SEVERITY_ICONS = {
  critical: '🚨',
  high: '⚠️',
  medium: '⚡',
  low: 'ℹ️',
}

export default function RiskReviewDashboard({ documents, token, apiUrl, onOpenUpload }) {
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
      setError('Please select a contract document to review.')
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
        throw new Error(data.detail || 'Failed to analyze contract risks.')
      }

      const data = await res.json()
      setReport(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
            <span>🛡️</span> Contract Risk Review & Audit
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            Automated compliance scanner combining deterministic business rules with legal LLM risk evaluation.
          </p>
        </div>
      </div>

      {/* Contract Selector */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-end gap-4">
        <div className="flex-1 w-full">
          <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
            Select Contract Document for Audit
          </label>
          <select
            value={selectedDoc}
            onChange={(e) => setSelectedDoc(e.target.value)}
            className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                📄 {d.filename} ({d.file_type.toUpperCase()})
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleReview}
          disabled={loading}
          className="px-6 py-2.5 bg-gradient-to-r from-red-600 via-rose-600 to-amber-600 hover:from-red-700 hover:to-amber-700 text-white font-bold rounded-xl shadow-md transition disabled:opacity-50 flex items-center gap-2 h-[42px]"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Auditing Rules & Context...
            </>
          ) : (
            <>⚡ Audit Contract Risks</>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 text-red-700 border border-red-200 rounded-xl text-xs font-medium">
          ⚠️ {error}
        </div>
      )}

      {/* Report Dashboard */}
      {report && (
        <div className="space-y-6 animate-fadeIn">
          {/* Overall Rating Banner */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-slate-100 flex items-center justify-center text-3xl shadow-inner">
                {SEVERITY_ICONS[report.overall_risk_level] || 'ℹ️'}
              </div>
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Overall Document Risk Level
                </div>
                <div className="text-2xl font-black text-slate-800 uppercase tracking-tight flex items-center gap-2">
                  <span
                    className={`px-3 py-0.5 rounded-full text-xs font-bold border ${
                      SEVERITY_COLORS[report.overall_risk_level] || 'bg-slate-100'
                    }`}
                  >
                    {report.overall_risk_level}
                  </span>
                  <span>Risk Profile</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 text-xs text-slate-600">
              <div className="text-center">
                <div className="text-xl font-bold text-slate-800">{report.total_risks_detected}</div>
                <div className="text-slate-400 font-medium">Risks Flagged</div>
              </div>
              <div className="h-8 w-px bg-slate-200"></div>
              <div className="text-center">
                <div className="text-xl font-bold text-slate-800">{Math.round(report.stats?.total_ms || 0)}ms</div>
                <div className="text-slate-400 font-medium">Scan Time</div>
              </div>
            </div>
          </div>

          {/* Risk Findings List */}
          <div className="space-y-4">
            <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <span>📋</span> Identified Risk Clauses ({report.findings?.length || 0})
            </h3>

            {report.findings?.length === 0 ? (
              <div className="bg-emerald-50 border border-emerald-200 p-8 rounded-2xl text-center text-emerald-800">
                <div className="text-3xl mb-2">🎉</div>
                <div className="font-bold text-base">No Critical Risks Detected</div>
                <p className="text-xs text-emerald-600 mt-1">
                  The scanned clauses comply with configured standard liability, termination, and indemnification policies.
                </p>
              </div>
            ) : (
              report.findings?.map((f, idx) => (
                <div
                  key={idx}
                  className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
                >
                  <div className="px-6 py-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2.5 py-0.5 rounded-md text-xs font-bold uppercase border ${
                          SEVERITY_COLORS[f.severity] || 'bg-slate-100'
                        }`}
                      >
                        {SEVERITY_ICONS[f.severity]} {f.severity}
                      </span>
                      <span className="font-bold text-sm text-slate-800">{f.rule_name}</span>
                    </div>

                    <span className="text-xs text-slate-400 font-mono">
                      Rule ID: {f.rule_id}
                    </span>
                  </div>

                  <div className="p-6 space-y-4">
                    {/* Clause snippet */}
                    <div>
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">
                        Flagged Clause Text:
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs font-serif text-slate-800 leading-relaxed italic">
                        "{f.clause_text}"
                      </div>
                    </div>

                    {/* Legal Explanation */}
                    <div>
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1">
                        Risk Analysis & Exposure:
                      </div>
                      <p className="text-xs text-slate-700 leading-relaxed font-sans">
                        {f.risk_explanation}
                      </p>
                    </div>

                    {/* Redline Recommendation */}
                    <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200/80">
                      <div className="text-xs font-bold text-amber-900 uppercase tracking-wider flex items-center gap-1.5 mb-1">
                        <span>💡</span> Actionable Negotiation / Redline Recommendation:
                      </div>
                      <p className="text-xs text-amber-800 leading-relaxed font-medium">
                        {f.recommendation}
                      </p>
                    </div>

                    {/* Citations */}
                    {f.citations?.length > 0 && (
                      <div className="flex flex-wrap gap-2 pt-1 border-t border-slate-100">
                        {f.citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            onClick={() => setActiveCitation(c)}
                            className="px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition border border-indigo-200 flex items-center gap-1.5"
                          >
                            <span>📌</span> Page {c.page} Source Evidence
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Citation Modal */}
      {activeCitation && (
        <CitationViewerModal citation={activeCitation} onClose={() => setActiveCitation(null)} />
      )}
    </div>
  )
}
