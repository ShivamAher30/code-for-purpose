import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Code2, Brain, FileText, Database, Eye } from 'lucide-react';

export default function TrustLayer({ trustLayer }) {
  const [expanded, setExpanded] = useState(false);

  if (!trustLayer || Object.keys(trustLayer).length <= 1) return null;

  const hasCode = trustLayer.pandas_code;
  const hasExplanation = trustLayer.explanation;
  const hasSources = trustLayer.sources?.length > 0;
  const hasAnalysis = trustLayer.analysis_data;

  return (
    <div className="animate-fade-in mt-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`
          flex items-center gap-2 text-[11px] font-medium transition-all duration-200
          px-3 py-1.5 rounded-lg
          ${expanded
            ? 'bg-surfaceHigh text-primarySoft'
            : 'text-textDim hover:text-textMuted hover:bg-surfaceContainer/40'
          }
        `}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Eye size={11} />
        <span>Transparency</span>
        {trustLayer.intent && (
          <span className="ml-1 text-[9px] font-medium text-textDim bg-surfaceHighest px-2 py-0.5 rounded-full">
            {trustLayer.intent.replace('_', ' ')}
          </span>
        )}
      </button>

      {expanded && (
        <div className="mt-2 space-y-3 pl-3 border-l-2 border-primary/10 animate-fade-in">
          {/* Generated Code */}
          {hasCode && (
            <div>
              <div className="flex items-center gap-1.5 text-[10px] font-semibold text-tertiarySoft/70 mb-1.5 uppercase tracking-[0.08em]">
                <Code2 size={11} />
                Generated Code
              </div>
              <pre className="bg-surfaceLowest rounded-xl p-3 text-[11px] font-mono text-primaryDim overflow-x-auto border border-outline/8">
                <code>{trustLayer.pandas_code}</code>
              </pre>
            </div>
          )}

          {/* Explanation */}
          {hasExplanation && (
            <div>
              <div className="flex items-center gap-1.5 text-[10px] font-semibold text-tertiarySoft/70 mb-1.5 uppercase tracking-[0.08em]">
                <Brain size={11} />
                Reasoning
              </div>
              <p className="text-[11px] text-textMuted/90 leading-relaxed bg-surfaceContainer/40 rounded-xl p-3">
                {trustLayer.explanation}
              </p>
            </div>
          )}

          {/* Sources */}
          {hasSources && (
            <div>
              <div className="flex items-center gap-1.5 text-[10px] font-semibold text-tertiarySoft/70 mb-1.5 uppercase tracking-[0.08em]">
                <FileText size={11} />
                Source Chunks
              </div>
              <div className="space-y-1.5">
                {trustLayer.sources.map((chunk, i) => (
                  <div key={i} className="text-[11px] text-textDim bg-surfaceContainer/40 rounded-xl p-2.5">
                    <span className="text-primarySoft/70 font-semibold text-[10px]">Chunk {i + 1}:</span>{' '}
                    {chunk.length > 200 ? chunk.slice(0, 200) + '…' : chunk}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw Analysis Data */}
          {hasAnalysis && (
            <details className="text-[11px]">
              <summary className="flex items-center gap-1.5 font-semibold text-textDim/50 cursor-pointer hover:text-textDim/70 transition-colors text-[10px] uppercase tracking-[0.08em]">
                <Database size={11} />
                Raw Analysis Data
              </summary>
              <pre className="mt-1.5 bg-surfaceLowest rounded-xl p-3 text-[10px] font-mono text-textDim/60 overflow-x-auto border border-outline/8 max-h-44 overflow-y-auto">
                {JSON.stringify(trustLayer.analysis_data, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
