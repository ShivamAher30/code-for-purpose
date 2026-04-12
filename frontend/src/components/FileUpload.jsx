import React, { useRef, useCallback, useState } from 'react';
import {
  UploadCloud, FileSpreadsheet, FileAudio, FileImage, FileText,
  File, CheckCircle2, Loader2,
} from 'lucide-react';

const FILE_CATEGORIES = {
  data: { exts: ['.csv', '.xlsx', '.xls'], label: 'Data', icon: FileSpreadsheet, color: '#818cf8' },
  audio: { exts: ['.mp3', '.wav', '.m4a', '.ogg', '.webm'], label: 'Audio', icon: FileAudio, color: '#a78bfa' },
  pdf: { exts: ['.pdf'], label: 'PDF', icon: FileText, color: '#f87171' },
  text: { exts: ['.txt', '.md', '.log'], label: 'Text', icon: File, color: '#34d399' },
};

function detectCategory(filename) {
  const ext = '.' + filename.split('.').pop().toLowerCase();
  for (const [cat, cfg] of Object.entries(FILE_CATEGORIES)) {
    if (cfg.exts.includes(ext)) return cat;
  }
  return null;
}

const ALL_ACCEPT = Object.values(FILE_CATEGORIES).flatMap(c => c.exts).join(',');

export default function FileUpload({ onUpload, onRagUpload, isUploading, uploadResult, ragResults }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file) return;
    const category = detectCategory(file.name);
    if (!category) {
      alert('Unsupported file type. Supported: CSV, Excel, PDF, Audio, Images, Text.');
      return;
    }
    if (category === 'data') {
      onUpload(file);
    } else {
      onRagUpload(file, category);
    }
  }, [onUpload, onRagUpload]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    files.forEach(f => handleFile(f));
  }, [handleFile]);

  return (
    <div className="sidebar-section">
      <p className="text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] mb-3 px-0.5">
        Data Sources
      </p>

      {/* Drop Zone */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`upload-zone rounded-xl p-4 cursor-pointer flex flex-col items-center gap-2.5 text-center
          ${dragOver ? 'drag-over' : ''}
          ${isUploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        {isUploading ? (
          <>
            <Loader2 size={20} className="text-primary animate-spin" />
            <p className="text-[11px] text-textMuted">Processing…</p>
          </>
        ) : (
          <>
            <div className="w-9 h-9 rounded-lg bg-primary/8 flex items-center justify-center">
              <UploadCloud size={18} className={`${dragOver ? 'text-primary' : 'text-textDim'} transition-colors`} />
            </div>
            <div>
              <p className="text-[11px] text-textMuted">
                <span className="text-primarySoft font-medium">Click to upload</span> or drag & drop
              </p>
              <div className="flex items-center justify-center gap-1.5 mt-1.5 flex-wrap">
                {Object.values(FILE_CATEGORIES).map(({ label, color }) => (
                  <span
                    key={label}
                    className="text-[9px] font-medium px-1.5 py-0.5 rounded-full"
                    style={{
                      color,
                      background: `${color}0d`,
                    }}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept={ALL_ACCEPT}
          onChange={(e) => handleFile(e.target.files[0])}
          className="hidden"
        />
      </div>

      {/* Data Upload Result */}
      {uploadResult && (
        <div className="animate-fade-in mt-3 rounded-xl bg-success/5 p-3 flex items-start gap-2.5">
          <CheckCircle2 size={14} className="text-success mt-0.5 shrink-0" />
          <div className="text-[11px] space-y-0.5">
            <p className="text-success font-medium">{uploadResult.rows?.toLocaleString()} rows loaded</p>
            <p className="text-textDim">{uploadResult.columns} columns · <span className="text-textMuted">{uploadResult.filename}</span></p>
          </div>
        </div>
      )}

      {/* RAG Upload Results */}
      {ragResults && ragResults.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {ragResults.map((r, i) => {
            const cat = FILE_CATEGORIES[r.type] || FILE_CATEGORIES.text;
            const Icon = cat.icon;
            return (
              <div key={i} className="animate-fade-in rounded-lg bg-surfaceContainer/50 p-2.5 flex items-start gap-2">
                <div
                  className="w-6 h-6 rounded-md flex items-center justify-center shrink-0"
                  style={{ background: `${cat.color}0d` }}
                >
                  <Icon size={12} style={{ color: cat.color }} />
                </div>
                <div className="text-[10px] space-y-0.5 min-w-0">
                  <p className="text-textMain font-medium truncate">{r.filename}</p>
                  {r.type === 'audio' && r.chunks_indexed && (
                    <p className="text-textDim">{r.chunks_indexed} chunks indexed</p>
                  )}
                  {r.type === 'pdf' && (
                    <p className="text-textDim">{r.pages} pages · {r.chunks_indexed} chunks{r.images_indexed > 0 ? ` · ${r.images_indexed} images` : ''}</p>
                  )}
                  {r.type === 'image' && (
                    <p className="text-textDim">Indexed via CLIP</p>
                  )}
                  {r.type === 'text' && (
                    <p className="text-textDim">Text indexed</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
