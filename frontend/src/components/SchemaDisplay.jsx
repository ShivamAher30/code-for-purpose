import React, { useState } from 'react';
import { Hash, Tag, Calendar, AlertTriangle, ChevronDown, ChevronRight, Columns3 } from 'lucide-react';

const badgeConfig = {
  numeric: { icon: Hash, color: '#818cf8', bg: '#818cf80d' },
  categorical: { icon: Tag, color: '#a78bfa', bg: '#a78bfa0d' },
  date: { icon: Calendar, color: '#67e8f9', bg: '#67e8f90d' },
  sensitive: { icon: AlertTriangle, color: '#f59e0b', bg: '#f59e0b0d' },
};

function Badge({ type, label }) {
  const cfg = badgeConfig[type] || badgeConfig.numeric;
  const Icon = cfg.icon;
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-medium"
      style={{ color: cfg.color, background: cfg.bg }}
    >
      <Icon size={9} />
      {label}
    </span>
  );
}

export default function SchemaDisplay({ schema }) {
  const [expanded, setExpanded] = useState(false);

  if (!schema) return null;

  const stats = [
    { label: 'Rows', value: schema.total_rows?.toLocaleString(), color: '#818cf8' },
    { label: 'Cols', value: schema.total_columns, color: '#a78bfa' },
    { label: 'Dates', value: schema.date_columns?.length || 0, color: '#67e8f9' },
  ];

  return (
    <div className="sidebar-section">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-[10px] font-semibold text-textDim uppercase tracking-[0.1em] hover:text-textMuted transition-colors px-0.5"
      >
        <span className="flex items-center gap-1.5">
          <Columns3 size={11} />
          Schema
        </span>
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>

      {/* Stat Cards */}
      <div className="grid grid-cols-3 gap-1.5 mt-3">
        {stats.map(({ label, value, color }) => (
          <div key={label} className="stat-card">
            <p className="text-sm font-bold tabular-nums" style={{ color }}>{value}</p>
            <p className="text-[8px] text-textDim uppercase tracking-wider mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Column Badges */}
      {expanded && (
        <div className="animate-fade-in space-y-2.5 max-h-44 overflow-y-auto pr-1 mt-3">
          {schema.numeric_columns?.length > 0 && (
            <div>
              <p className="text-[9px] text-textDim/60 font-medium mb-1 px-0.5">Numeric</p>
              <div className="flex flex-wrap gap-1">
                {schema.numeric_columns.map(col => (
                  <Badge key={col} type="numeric" label={col} />
                ))}
              </div>
            </div>
          )}
          {schema.categorical_columns?.length > 0 && (
            <div>
              <p className="text-[9px] text-textDim/60 font-medium mb-1 px-0.5">Categorical</p>
              <div className="flex flex-wrap gap-1">
                {schema.categorical_columns.map(col => (
                  <Badge key={col} type="categorical" label={col} />
                ))}
              </div>
            </div>
          )}
          {schema.date_columns?.length > 0 && (
            <div>
              <p className="text-[9px] text-textDim/60 font-medium mb-1 px-0.5">Date</p>
              <div className="flex flex-wrap gap-1">
                {schema.date_columns.map(col => (
                  <Badge key={col} type="date" label={col} />
                ))}
              </div>
            </div>
          )}
          {schema.sensitive_columns?.length > 0 && (
            <div>
              <p className="text-[9px] text-textDim/60 font-medium mb-1 px-0.5">Sensitive</p>
              <div className="flex flex-wrap gap-1">
                {schema.sensitive_columns.map(s => (
                  <Badge key={s.name} type="sensitive" label={s.name} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
