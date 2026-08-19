import { useState, useEffect } from 'react'
import CitationViewerModal from './CitationViewerModal'

// Lightweight Markdown renderer: handles bold, headings, tables, hr, lists
function renderMarkdown(text) {
  if (!text) return ''
  const lines = text.split('\n')
  const result = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    // Table detection: starts with |
    if (line.trim().startsWith('|')) {
      const tableLines = []
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        tableLines.push(lines[i])
        i++
      }
      // skip separator row (|:---|:---|)
      const rows = tableLines.filter(l => !l.match(/^\|[-:\s|]+\|?\s*$/))
      const tableHtml = rows.map((row, rIdx) => {
        const cells = row.split('|').filter((_, ci) => ci > 0 && ci < row.split('|').length - 1)
        const tag = rIdx === 0 ? 'th' : 'td'
        return `<tr>${cells.map(c => `<${tag} class="border border-indigo-800/30 px-3 py-2 text-left text-xs ${rIdx === 0 ? 'font-bold text-indigo-200 bg-indigo-900/40' : 'text-slate-200'}">${inlineFormat(c.trim())}</${tag}>`).join('')}</tr>`
      }).join('')
      result.push(`<div class="overflow-x-auto my-3"><table class="w-full border-collapse text-xs">${tableHtml}</table></div>`)
      continue
    }
    // Horizontal rule
    if (line.trim() === '---' || line.trim() === '***') {
      result.push('<hr class="border-indigo-800/40 my-3" />')
      i++
      continue
    }
    // Headings
    if (line.startsWith('#### ')) {
      result.push(`<h4 class="text-xs font-bold text-indigo-200 mt-4 mb-1">${inlineFormat(line.slice(5))}</h4>`)
      i++
      continue
    }
    if (line.startsWith('### ')) {
      result.push(`<h3 class="text-sm font-bold text-white mt-5 mb-1">${inlineFormat(line.slice(4))}</h3>`)
      i++
      continue
    }
    if (line.startsWith('## ')) {
      result.push(`<h2 class="text-base font-bold text-white mt-5 mb-1">${inlineFormat(line.slice(3))}</h2>`)
      i++
      continue
    }
    // Bullet list
    if (line.trim().startsWith('* ') || line.trim().startsWith('- ')) {
      result.push(`<li class="ml-4 text-xs text-slate-300 list-disc list-inside">${inlineFormat(line.trim().slice(2))}</li>`)
      i++
      continue
    }
    // Empty line
    if (line.trim() === '') {
      result.push('<div class="h-2"></div>')
      i++
      continue
    }
    // Normal paragraph
    result.push(`<p class="text-xs text-slate-200 leading-relaxed">${inlineFormat(line)}</p>`)
    i++
  }
  return result.join('')
}

function inlineFormat(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white">$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code class="bg-indigo-900/50 px-1 rounded text-indigo-200">$1</code>')
}

const DEFAULT_FACETS = [
  'Thời hạn hợp đồng & Quyền chấm dứt (Term & Termination)',
  'Quy định thời hạn báo trước (Notice Period Requirements)',
  'Giới hạn trách nhiệm & Mức trần bồi thường (Limitation of Liability)',
  'Nghĩa vụ bồi thường bên thứ ba (Indemnification)',
  'Luật áp dụng & Cơ quan giải quyết tranh chấp (Governing Law & Forum)',
  'Điều khoản thanh toán & Điều chỉnh giá (Payment & Pricing)',
  'Bảo mật thông tin & Dữ liệu (Confidentiality & Data Protection)',
]

export default function ContractCompareView({ documents, token, apiUrl, onOpenUpload }) {
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
    if (selectedFacets.includes(facet)) {
      setSelectedFacets(selectedFacets.filter((f) => f !== facet))
    } else {
      setSelectedFacets([...selectedFacets, facet])
    }
  }

  const handleCompare = async () => {
    if (!contractA || !contractB) {
      setError('Please select two contracts to compare.')
      return
    }
    if (contractA === contractB) {
      setError('Please select two different contracts for comparison.')
      return
    }
    if (selectedFacets.length === 0) {
      setError('Please select at least one legal facet to compare.')
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
          contract_a_id: contractA,
          contract_b_id: contractB,
          facets: selectedFacets,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to compare contracts.')
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto w-full space-y-6">
      {/* View Header */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
            <span>⚖️</span> Multi-Contract Comparison
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            Perform facet-decomposed independent retrieval and side-by-side clause contrast across two agreements.
          </p>
        </div>
      </div>

      {/* Contract Selection & Facets Picker */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Select Contract A
            </label>
            <select
              value={contractA}
              onChange={(e) => setContractA(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              {documents.map((d) => (
                <option key={d.id} value={d.id}>
                  📄 {d.filename} ({d.file_type.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Select Contract B
            </label>
            <select
              value={contractB}
              onChange={(e) => setContractB(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              {documents.map((d) => (
                <option key={d.id} value={d.id}>
                  📄 {d.filename} ({d.file_type.toUpperCase()})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Facet Checkboxes */}
        <div>
          <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2.5">
            Comparison Facets to Retrieve & Contrast:
          </label>
          <div className="flex flex-wrap gap-2">
            {DEFAULT_FACETS.map((facet) => {
              const active = selectedFacets.includes(facet)
              return (
                <button
                  key={facet}
                  type="button"
                  onClick={() => toggleFacet(facet)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border ${
                    active
                      ? 'bg-indigo-50 border-indigo-300 text-indigo-700 shadow-xs'
                      : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
                  }`}
                >
                  {active ? '✓ ' : '+ '}
                  {facet}
                </button>
              )
            })}
          </div>
        </div>

        {error && (
          <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-xl text-xs font-medium">
            ⚠️ {error}
          </div>
        )}

        <button
          onClick={handleCompare}
          disabled={loading}
          className="w-full md:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white font-bold rounded-xl shadow-md transition disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Decomposing & Retrieving Evidence...
            </>
          ) : (
            <>🔍 Run Comparison Analysis</>
          )}
        </button>
      </div>

      {/* Comparison Results */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Executive Summary Banner */}
          <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-950 p-6 rounded-2xl text-white shadow-lg border border-indigo-900/50">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
                Executive Legal Summary
              </span>
              <span className="text-xs bg-indigo-500/20 text-indigo-200 px-2.5 py-1 rounded-full border border-indigo-400/30">
                ⚡ Processed in {Math.round(result.stats?.total_ms || 0)}ms
              </span>
            </div>
            <div
              className="text-sm leading-relaxed font-sans"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(result.summary_comparison) }}
            />
          </div>

          {/* Facet Comparison Cards */}
          <div className="space-y-4">
            <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <span>📋</span> Facet-by-Facet Contrast ({result.facet_comparisons?.length || 0} Facets)
            </h3>

            {result.facet_comparisons?.map((facet, idx) => (
              <div
                key={idx}
                className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden"
              >
                <div className="px-6 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <span className="font-bold text-sm text-slate-800 flex items-center gap-2">
                    <span className="w-5 h-5 rounded-md bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-bold">
                      {idx + 1}
                    </span>
                    {facet.facet_name}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-100 p-6 gap-6">
                  {/* Contract A */}
                  <div className="space-y-3">
                    <div className="text-xs font-bold text-blue-700 uppercase tracking-wider flex items-center gap-1.5">
                      <span>📄</span> {result.contract_a_name}
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed bg-blue-50/40 p-3.5 rounded-xl border border-blue-100">
                      {facet.contract_a_findings}
                    </p>
                    {facet.contract_a_citations?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {facet.contract_a_citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            onClick={() => setActiveCitation(c)}
                            className="px-2.5 py-1 text-[11px] font-medium rounded-lg bg-blue-100/70 text-blue-800 hover:bg-blue-200 transition border border-blue-200 flex items-center gap-1"
                          >
                            <span>📌</span> P.{c.page} {c.section_path?.[0] || 'Clause'}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Contract B */}
                  <div className="space-y-3">
                    <div className="text-xs font-bold text-indigo-700 uppercase tracking-wider flex items-center gap-1.5">
                      <span>📄</span> {result.contract_b_name}
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed bg-indigo-50/40 p-3.5 rounded-xl border border-indigo-100">
                      {facet.contract_b_findings}
                    </p>
                    {facet.contract_b_citations?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {facet.contract_b_citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            onClick={() => setActiveCitation(c)}
                            className="px-2.5 py-1 text-[11px] font-medium rounded-lg bg-indigo-100/70 text-indigo-800 hover:bg-indigo-200 transition border border-indigo-200 flex items-center gap-1"
                          >
                            <span>📌</span> P.{c.page} {c.section_path?.[0] || 'Clause'}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Key Differences & Risk Assessment Strip */}
                <div className="px-6 py-3 bg-amber-50/50 border-t border-amber-100/80 flex flex-col md:flex-row md:items-center justify-between text-xs gap-2">
                  <div className="text-slate-700">
                    <strong className="text-amber-800">⚡ Key Discrepancy:</strong> {facet.key_differences}
                  </div>
                  {facet.risk_assessment && (
                    <div className="text-amber-900 font-medium bg-amber-100 px-2.5 py-0.5 rounded-md self-start md:self-auto">
                      {facet.risk_assessment}
                    </div>
                  )}
                </div>
              </div>
            ))}
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
