import { useState } from 'react'

export default function DebugPanel({ stats, onClose }) {
  if (!stats) return null

  const total = stats.total_ms || 1.0
  const routingPct = Math.min(100, Math.round(((stats.routing_ms || 0) / total) * 100))
  const retrievalPct = Math.min(100, Math.round(((stats.retrieval_ms || 0) / total) * 100))
  const rerankPct = Math.min(100, Math.round(((stats.rerank_ms || 0) / total) * 100))
  const genPct = Math.min(100, Math.round(((stats.generation_ms || 0) / total) * 100))
  const verPct = Math.min(100, Math.round(((stats.verification_ms || 0) / total) * 100))

  return (
    <div className="fixed bottom-4 right-4 z-40 bg-slate-900 text-white rounded-2xl shadow-2xl border border-slate-700 p-5 max-w-sm w-full animate-scaleUp">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
        <div className="flex items-center gap-2 font-bold text-xs uppercase tracking-wider text-indigo-400">
          <span>⚡</span> Query Observability & Telemetry
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white text-lg leading-none"
        >
          &times;
        </button>
      </div>

      <div className="space-y-3.5 text-xs">
        {/* Retrieval Path & Confidence */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400">Adaptive Path:</span>
          <span className="font-semibold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            {stats.retrieval_path || 'Fast Hybrid'}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-400">Confidence Score:</span>
          <div className="flex items-center gap-2">
            <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-emerald-400 h-full rounded-full"
                style={{ width: `${Math.round((stats.confidence_score || 1.0) * 100)}%` }}
              ></div>
            </div>
            <span className="font-mono text-emerald-400 font-bold">
              {Math.round((stats.confidence_score || 1.0) * 100)}%
            </span>
          </div>
        </div>

        {/* Latency Breakdown Bar */}
        <div>
          <div className="flex justify-between text-slate-400 text-[11px] mb-1">
            <span>Latency Breakdown ({Math.round(total)}ms)</span>
          </div>
          <div className="h-2 w-full bg-slate-800 rounded-full flex overflow-hidden">
            <div style={{ width: `${routingPct}%` }} className="bg-amber-400" title="Routing"></div>
            <div style={{ width: `${retrievalPct}%` }} className="bg-blue-400" title="Retrieval"></div>
            <div style={{ width: `${rerankPct}%` }} className="bg-purple-400" title="Reranker"></div>
            <div style={{ width: `${genPct}%` }} className="bg-indigo-500" title="Generation"></div>
            <div style={{ width: `${verPct}%` }} className="bg-emerald-400" title="Verification"></div>
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>Ret: {Math.round(stats.retrieval_ms || 0)}ms</span>
            <span>Gen: {Math.round(stats.generation_ms || 0)}ms</span>
            <span>Ver: {Math.round(stats.verification_ms || 0)}ms</span>
          </div>
        </div>

        {/* LLM Calls & Tokens */}
        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800 text-[11px]">
          <div className="bg-slate-800/60 p-2 rounded-lg text-center">
            <div className="text-slate-400">LLM Calls</div>
            <div className="font-bold text-white text-sm">{stats.llm_calls_count || 1}</div>
          </div>
          <div className="bg-slate-800/60 p-2 rounded-lg text-center">
            <div className="text-slate-400">Est. Tokens</div>
            <div className="font-bold text-white text-sm">{stats.estimated_tokens || '~450'}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
