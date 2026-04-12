import React, { useState } from 'react';
import { ChevronDown, ChevronRight, ArrowUpDown, Table2 } from 'lucide-react';

function formatCell(value) {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (!isNaN(num) && value !== '' && typeof value !== 'boolean') {
    if (Number.isInteger(num)) return num.toLocaleString();
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return String(value);
}

export default function DataTable({ data, title }) {
  const [expanded, setExpanded] = useState(true);
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState('asc');

  if (!data || !data.columns?.length) return null;

  const handleSort = (colIdx) => {
    if (sortCol === colIdx) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCol(colIdx);
      setSortDir('asc');
    }
  };

  let rows = [...data.rows];
  if (sortCol !== null) {
    rows.sort((a, b) => {
      const va = a[sortCol];
      const vb = b[sortCol];
      const na = parseFloat(va);
      const nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) {
        return sortDir === 'asc' ? na - nb : nb - na;
      }
      return sortDir === 'asc'
        ? String(va).localeCompare(String(vb))
        : String(vb).localeCompare(String(va));
    });
  }

  return (
    <div className="animate-fade-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-[11px] font-medium text-textDim hover:text-textMuted transition-colors mb-2"
      >
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <Table2 size={12} className="text-primary/50" />
        <span>{title || 'Data'}</span>
        <span className="text-textDim/60 font-normal tabular-nums">
          ({data.shown_rows} of {data.total_rows?.toLocaleString()})
        </span>
      </button>

      {expanded && (
        <div className="rounded-xl overflow-hidden bg-surfaceLowest">
          <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 z-10">
                <tr className="bg-surfaceHigh">
                  {data.columns.map((col, i) => (
                    <th
                      key={i}
                      onClick={() => handleSort(i)}
                      className="px-3 py-2.5 text-left font-semibold text-primarySoft/80 cursor-pointer
                        hover:bg-surfaceHighest transition-colors whitespace-nowrap select-none
                        border-b border-outline/10 first:pl-4 last:pr-4"
                    >
                      <span className="inline-flex items-center gap-1.5">
                        {col}
                        <ArrowUpDown
                          size={9}
                          className={`transition-colors ${sortCol === i ? 'text-primary' : 'text-textDim/25'}`}
                        />
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr
                    key={ri}
                    className={`
                      transition-colors
                      ${ri % 2 === 0 ? 'bg-surfaceLow/50' : 'bg-transparent'}
                      hover:bg-surfaceContainer/60
                    `}
                  >
                    {row.map((cell, ci) => (
                      <td
                        key={ci}
                        className="px-3 py-2 text-textMuted/90 whitespace-nowrap max-w-[220px] truncate
                          tabular-nums first:pl-4 last:pr-4"
                      >
                        {formatCell(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
