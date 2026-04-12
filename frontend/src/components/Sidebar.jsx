import React, { useState } from 'react';
import {
  Database, FileBarChart, AlertCircle, Lightbulb, Trash2,
  Settings, Shield, ChevronDown, ChevronRight,
  Download, FileDown, Music, Image, FileText,
} from 'lucide-react';
import FileUpload from './FileUpload';
import SchemaDisplay from './SchemaDisplay';

export default function Sidebar({
  schema,
  onUpload,
  onRagUpload,
  isUploading,
  uploadResult,
  ragResults,
  ragFiles,
  onQuickAction,
  isProcessing,
  routingMode,
  setRoutingMode,
  onClearChat,
  onClearRag,
  onExportCSV,
  onExportPDF,
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [ragFilesOpen, setRagFilesOpen] = useState(false);

  const quickActions = [
    { id: 'summary', label: 'Summary', icon: FileBarChart, color: '#818cf8' },
    { id: 'anomaly', label: 'Anomalies', icon: AlertCircle, color: '#f87171' },
    { id: 'insights', label: 'Insights', icon: Lightbulb, color: '#f59e0b' },
  ];

  const routingOptions = [
    { value: 'Auto-Detect', label: 'Auto', desc: 'AI picks best mode' },
    { value: 'Structured (CSV)', label: 'Structured', desc: 'CSV/Excel queries' },
    { value: 'Unstructured (RAG)', label: 'RAG', desc: 'Document search' },
  ];

  const hasRagFiles = ragFiles && (ragFiles.audio?.length > 0 || ragFiles.images?.length > 0 || ragFiles.has_text_index);

  return (
    <aside className="w-[272px] bg-surfaceLowest flex flex-col h-full overflow-hidden shrink-0">
      {/* ── Logo ── */}
      <div className="p-5 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-tertiary flex items-center justify-center">
            <Database size={16} className="text-white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="font-display font-bold text-[14px] text-textMain tracking-tight leading-tight">
              Talk-to-Data<span className="text-tertiary">.AI</span>
            </h1>
            <p className="text-[10px] text-textDim font-medium">Intelligent Analytics</p>
          </div>
        </div>
      </div>

      {/* ── Scrollable Content ── */}
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-1">

        {/* File Upload */}
        <FileUpload
          onUpload={onUpload}
          onRagUpload={onRagUpload}
          isUploading={isUploading}
          uploadResult={uploadResult}
          ragResults={ragResults}
        />

        {/* Schema */}
        {schema && <SchemaDisplay schema={schema} />}

        {/* Quick Actions */}
        {schema && (
          <div className="sidebar-section">
            <p className="text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] mb-3 px-0.5">
              Quick Actions
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {quickActions.map(({ id, label, icon: Icon, color }) => (
                <button
                  key={id}
                  onClick={() => onQuickAction(id)}
                  disabled={isProcessing}
                  className="quick-action-btn"
                >
                  <Icon size={13} style={{ color }} />
                  <span>{label}</span>
                </button>
              ))}
              <button
                onClick={onClearChat}
                className="quick-action-btn hover:!bg-error/8 hover:!text-error"
              >
                <Trash2 size={13} />
                <span>Clear</span>
              </button>
            </div>
          </div>
        )}

        {/* RAG Knowledge Base */}
        {hasRagFiles && (
          <div className="sidebar-section">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setRagFilesOpen(!ragFilesOpen)}
                className="flex items-center justify-between flex-1 text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] hover:text-textMuted transition-colors px-0.5"
              >
                <span className="flex items-center gap-1.5">
                  <Database size={11} />
                  Knowledge Base
                </span>
                {ragFilesOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              </button>
              <button
                onClick={onClearRag}
                title="Clear all RAG data"
                className="p-1.5 rounded-lg text-textDim hover:text-error hover:bg-error/8 transition-all duration-200"
              >
                <Trash2 size={11} />
              </button>
            </div>

            {ragFilesOpen && (
              <div className="animate-fade-in mt-2 space-y-1 max-h-32 overflow-y-auto">
                {ragFiles.audio?.map((f, i) => (
                  <div key={`a-${i}`} className="rag-chip">
                    <Music size={11} className="text-tertiary shrink-0" />
                    <span className="truncate">{f.name}</span>
                  </div>
                ))}
                {ragFiles.images?.slice(0, 6).map((f, i) => (
                  <div key={`i-${i}`} className="rag-chip">
                    <Image size={11} className="text-info shrink-0" />
                    <span className="truncate">{f.name}</span>
                  </div>
                ))}
                {ragFiles.has_text_index && (
                  <div className="rag-chip">
                    <FileText size={11} className="text-success shrink-0" />
                    <span>Text / PDF Index</span>
                  </div>
                )}
                {ragFiles.images?.length > 6 && (
                  <p className="text-[10px] text-textDim px-1">+{ragFiles.images.length - 6} more</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Export */}
        {schema && (
          <div className="sidebar-section">
            <p className="text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] mb-3 px-0.5">
              Export
            </p>
            <div className="flex gap-1.5">
              <button onClick={onExportCSV} className="quick-action-btn flex-1 justify-center">
                <Download size={12} />
                <span>CSV</span>
              </button>
              <button onClick={onExportPDF} className="quick-action-btn flex-1 justify-center">
                <FileDown size={12} />
                <span>PDF</span>
              </button>
            </div>
          </div>
        )}

        {/* Settings */}
        <div className="space-y-2">
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            className="flex items-center justify-between w-full text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] hover:text-textMuted transition-colors px-0.5"
          >
            <span className="flex items-center gap-1.5">
              <Settings size={11} />
              Settings
            </span>
            {settingsOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>

          {settingsOpen && (
            <div className="animate-fade-in space-y-3 mt-1">
              {/* Routing Mode */}
              <div className="space-y-1">
                <p className="text-[10px] text-textDim font-medium px-0.5">Routing Mode</p>
                <div className="space-y-0.5">
                  {routingOptions.map(({ value, label, desc }) => (
                    <button
                      key={value}
                      onClick={() => setRoutingMode(value)}
                      className={`
                        flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-[11px] transition-all duration-200
                        ${routingMode === value
                          ? 'bg-primary/8 text-primarySoft'
                          : 'text-textDim hover:text-textMuted hover:bg-surfaceContainer/50'
                        }
                      `}
                    >
                      <div className={`w-3 h-3 rounded-full border-2 flex items-center justify-center
                        ${routingMode === value ? 'border-primary' : 'border-outline/50'}
                      `}>
                        {routingMode === value && <div className="w-1.5 h-1.5 rounded-full bg-primary" />}
                      </div>
                      <div className="text-left">
                        <span className="font-medium">{label}</span>
                        <p className="text-[9px] text-textDim mt-0.5">{desc}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Privacy Toggle */}
              <div className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-surfaceContainer/40">
                <span className="flex items-center gap-1.5 text-[11px] text-textDim">
                  <Shield size={12} />
                  PII Masking
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" />
                  <div className="w-8 h-4 bg-surfaceHighest rounded-full peer peer-checked:bg-primary/50 transition-colors
                    after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-textDim after:rounded-full after:h-3 after:w-3 after:transition-all
                    peer-checked:after:translate-x-4 peer-checked:after:bg-primary"></div>
                </label>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="p-4 border-t border-outline/10">
        <p className="text-[9px] text-textDim text-center">
          Made for  <span className="gradient-text font-semibold">Code for Purpose </span> 
        </p>
      </div>
    </aside>
  );
}
