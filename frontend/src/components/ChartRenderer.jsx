import React, { useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine, Area, AreaChart, Legend,
  Brush, Rectangle,
} from 'recharts';

/* ── Color palette: curated, not random ── */
const CHART_COLORS = [
  '#818cf8', '#a78bfa', '#67e8f9', '#f59e0b', '#34d399',
  '#f472b6', '#60a5fa', '#fb923c', '#4ade80', '#e879f9',
];

const DARK = {
  bg: '#131b2e',
  grid: 'rgba(70,69,84,0.10)',
  text: '#908fa0',
  textBright: '#c7c4d7',
  label: '#dae2fd',
  axisLine: 'rgba(70,69,84,0.15)',
};

/* ── Number formatting: 1200 → 1.2K, 2500000 → 2.5M ── */
function formatNumber(value) {
  if (typeof value !== 'number' || isNaN(value)) return value;
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return (value / 1_000_000_000).toFixed(1).replace(/\.0$/, '') + 'B';
  if (abs >= 1_000_000) return (value / 1_000_000).toFixed(1).replace(/\.0$/, '') + 'M';
  if (abs >= 10_000) return (value / 1_000).toFixed(1).replace(/\.0$/, '') + 'K';
  if (abs >= 1_000) return (value / 1_000).toFixed(1).replace(/\.0$/, '') + 'K';
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

/* ── Full number for tooltips ── */
function formatFullNumber(value) {
  if (typeof value !== 'number' || isNaN(value)) return value;
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/* ── Truncate long axis labels ── */
function truncateLabel(label, maxLen = 14) {
  if (typeof label !== 'string') return label;
  return label.length > maxLen ? label.slice(0, maxLen) + '…' : label;
}

/* ── Smart angle for axis labels ── */
function needsAngle(data) {
  if (!data || data.length === 0) return false;
  const avgLen = data.reduce((sum, d) => sum + String(d.name || '').length, 0) / data.length;
  return data.length > 6 || avgLen > 8;
}

/* ── Custom Tooltip ── */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 mt-1">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ background: entry.color }}
          />
          <span className="value">
            <span style={{ color: DARK.textBright }}>{entry.name || 'Value'}:</span>{' '}
            {formatFullNumber(entry.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── Histogram Tooltip ── */
function HistogramTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">{d.range_start} – {d.range_end}</p>
      <div className="flex items-center gap-2 mt-1">
        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#818cf8' }} />
        <span className="value">
          <span style={{ color: DARK.textBright }}>Count:</span> {d.count?.toLocaleString()}
        </span>
      </div>
    </div>
  );
}

/* ── Heatmap Tooltip ── */
function HeatmapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">{d.x} × {d.y}</p>
      <div className="flex items-center gap-2 mt-1">
        <span className="value">
          <span style={{ color: DARK.textBright }}>Correlation:</span>{' '}
          <span style={{ color: d.value > 0.5 ? '#34d399' : d.value < -0.5 ? '#f87171' : DARK.text }}>
            {d.value?.toFixed(3)}
          </span>
        </span>
      </div>
    </div>
  );
}

/* ── Custom Pie Label ── */
function renderPieLabel({ name, percent }) {
  if (percent < 0.03) return null;
  return `${(percent * 100).toFixed(0)}%`;
}

/* ── Legend ── */
function CustomLegend({ payload }) {
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1.5 justify-center pt-3">
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-1.5 text-[11px]" style={{ color: DARK.textBright }}>
          <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: entry.color }} />
          <span>{entry.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Chart Wrapper with title and optional summary ── */
function ChartWrapper({ children, title, summary, height }) {
  return (
    <div className="w-full animate-fade-in">
      {title && (
        <div className="mb-4">
          <p className="text-[11px] text-textDim font-semibold uppercase tracking-[0.08em] flex items-center gap-2">
            <span className="w-1 h-3 rounded-full bg-primary/60" />
            {title}
          </p>
          {summary && (
            <p className="text-[11px] text-textMuted mt-1 pl-3">{summary}</p>
          )}
        </div>
      )}
      <div className="w-full" style={{ height: height || 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ── Axis defaults ── */
const xAxisDefaults = (data, customProps = {}) => {
  const angled = needsAngle(data);
  return {
    dataKey: 'name',
    stroke: 'none',
    tick: {
      fill: DARK.text,
      fontSize: 10,
      fontFamily: 'Inter, sans-serif',
    },
    axisLine: false,
    tickLine: false,
    tickMargin: angled ? 8 : 6,
    angle: angled ? -35 : 0,
    textAnchor: angled ? 'end' : 'middle',
    height: angled ? 60 : 30,
    tickFormatter: (v) => truncateLabel(v),
    ...customProps,
  };
};

const yAxisDefaults = {
  stroke: 'none',
  tick: {
    fill: DARK.text,
    fontSize: 10,
    fontFamily: 'Inter, sans-serif',
  },
  axisLine: false,
  tickLine: false,
  tickFormatter: formatNumber,
  width: 48,
};

const gridDefaults = {
  strokeDasharray: '3 3',
  stroke: DARK.grid,
  vertical: false,
};

const brushDefaults = {
  dataKey: 'name',
  height: 24,
  stroke: 'rgba(129,140,248,0.3)',
  fill: 'rgba(19,27,46,0.8)',
  tickFormatter: () => '',
};

export default function ChartRenderer({ data, chartType, chartKeys, title }) {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  if (!data || data.length === 0) return null;

  const keys = chartKeys || Object.keys(data[0]).filter(
    k => k !== 'name' && k !== 'index' && k !== 'is_anomaly' && k !== 'label'
      && k !== 'range_start' && k !== 'range_end' && k !== 'x' && k !== 'y'
      && k !== 'outliers' && k !== 'q1' && k !== 'q3' && k !== 'median'
      && k !== 'min' && k !== 'max' && k !== 'mean'
  );

  const showBrush = data.length > 15 && !['pie', 'heatmap', 'box', 'scatter_xy', 'anomaly_scatter'].includes(chartType);

  // ── Histogram ──
  if (chartType === 'histogram') {
    return (
      <ChartWrapper title={title}>
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <defs>
            <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#818cf8" stopOpacity={0.9} />
              <stop offset="100%" stopColor="#818cf8" stopOpacity={0.45} />
            </linearGradient>
          </defs>
          <CartesianGrid {...gridDefaults} />
          <XAxis {...xAxisDefaults(data)} />
          <YAxis {...yAxisDefaults} />
          <Tooltip content={<HistogramTooltip />} cursor={{ fill: 'rgba(129,140,248,0.04)' }} />
          <Bar dataKey="count" fill="url(#histGrad)" radius={[3, 3, 0, 0]} maxBarSize={50} />
          {showBrush && <Brush {...brushDefaults} />}
        </BarChart>
      </ChartWrapper>
    );
  }

  // ── Box Plot (custom) ──
  if (chartType === 'box') {
    const d = data[0];
    if (!d) return null;
    const allValues = [d.min, d.q1, d.median, d.q3, d.max, ...(d.outliers || [])];
    const boxMin = Math.min(...allValues);
    const boxMax = Math.max(...allValues);
    const padding = (boxMax - boxMin) * 0.1 || 1;

    return (
      <ChartWrapper title={title}>
        <BarChart
          data={[{
            name: d.name,
            bottom: d.q1,
            box: d.q3 - d.q1,
            median: d.median,
            min: d.min,
            max: d.max,
            mean: d.mean,
          }]}
          margin={{ top: 20, right: 40, left: 40, bottom: 20 }}
          layout="vertical"
        >
          <CartesianGrid {...gridDefaults} horizontal={false} vertical />
          <XAxis
            type="number"
            domain={[boxMin - padding, boxMax + padding]}
            stroke="none"
            tick={{ fill: DARK.text, fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatNumber}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="none"
            tick={{ fill: DARK.textBright, fontSize: 11, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            width={60}
          />
          {/* IQR Box */}
          <Bar dataKey="bottom" stackId="box" fill="transparent" barSize={32} />
          <Bar dataKey="box" stackId="box" fill="rgba(129,140,248,0.25)" stroke="#818cf8" strokeWidth={1.5} radius={[4, 4, 4, 4]} barSize={32} />
          {/* Whiskers and median line */}
          <ReferenceLine x={d.median} stroke="#818cf8" strokeWidth={2.5} strokeDasharray="" label={{ value: `Median: ${formatNumber(d.median)}`, position: 'top', fill: DARK.textBright, fontSize: 10 }} />
          <ReferenceLine x={d.min} stroke={DARK.text} strokeWidth={1} strokeDasharray="4 4" label={{ value: `Min: ${formatNumber(d.min)}`, position: 'bottom', fill: DARK.text, fontSize: 9 }} />
          <ReferenceLine x={d.max} stroke={DARK.text} strokeWidth={1} strokeDasharray="4 4" label={{ value: `Max: ${formatNumber(d.max)}`, position: 'bottom', fill: DARK.text, fontSize: 9 }} />
          {d.mean && (
            <ReferenceLine x={d.mean} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: `Mean: ${formatNumber(d.mean)}`, position: 'top', fill: '#f59e0b', fontSize: 9 }} />
          )}
          <Tooltip content={<BoxTooltip data={d} />} />
        </BarChart>
      </ChartWrapper>
    );
  }

  // ── Heatmap ──
  if (chartType === 'heatmap') {
    const heatCols = chartKeys || [...new Set(data.map(d => d.x))];
    const cellSize = Math.max(28, Math.min(48, 400 / heatCols.length));
    const size = heatCols.length * cellSize + 120; // Increased padding for labels

    return (
      <ChartWrapper title={title} height={Math.min(size + 80, 500)}>
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${size + 20} ${size + 20}`} style={{ overflow: 'visible' }}>
            {/* Column labels (top) */}
            {heatCols.map((col, i) => (
              <text
                key={`xt-${col}`}
                x={120 + i * cellSize + cellSize / 2}
                y={60}
                textAnchor="end"
                transform={`rotate(-45, ${120 + i * cellSize + cellSize / 2}, 60)`}
                fill={DARK.text}
                fontSize={9}
                fontFamily="Inter"
              >
                {truncateLabel(col, 10)}
              </text>
            ))}
            {/* Row labels (left) */}
            {heatCols.map((col, j) => (
              <text
                key={`yt-${col}`}
                x={116}
                y={80 + j * cellSize + cellSize / 2 + 4}
                textAnchor="end"
                fill={DARK.text}
                fontSize={9}
                fontFamily="Inter"
              >
                {truncateLabel(col, 10)}
              </text>
            ))}
            {/* Cells */}
            {data.map((cell, idx) => {
              const xi = heatCols.indexOf(cell.x);
              const yi = heatCols.indexOf(cell.y);
              if (xi === -1 || yi === -1) return null;
              const v = cell.value;
              const color = v > 0
                ? `rgba(52, 211, 153, ${Math.min(Math.abs(v), 1) * 0.8 + 0.1})`
                : v < 0
                ? `rgba(248, 113, 113, ${Math.min(Math.abs(v), 1) * 0.8 + 0.1})`
                : 'rgba(70,69,84,0.15)';

              return (
                <g key={idx}>
                  <rect
                    x={120 + xi * cellSize + 1}
                    y={80 + yi * cellSize + 1}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    rx={4}
                    fill={color}
                    style={{ transition: 'fill 0.2s', cursor: 'pointer' }}
                    onMouseEnter={(e) => {
                      setHoveredCell(cell);
                      setTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseMove={(e) => {
                      setTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                  </rect>
                  {cellSize >= 32 && (
                    <text
                      x={120 + xi * cellSize + cellSize / 2}
                      y={80 + yi * cellSize + cellSize / 2 + 4}
                      textAnchor="middle"
                      fill={Math.abs(v) > 0.4 ? '#fff' : DARK.text}
                      fontSize={8}
                      fontWeight={500}
                      fontFamily="Inter"
                      pointerEvents="none"
                    >
                      {v.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
          {hoveredCell && (
            <div style={{ position: 'fixed', left: tooltipPos.x + 15, top: tooltipPos.y + 15, pointerEvents: 'none', zIndex: 1000 }}>
              <HeatmapTooltip active={true} payload={[{ payload: hoveredCell }]} />
            </div>
          )}
        </div>
      </ChartWrapper>
    );
  }

  // ── Scatter XY ──
  if (chartType === 'scatter_xy') {
    const xKey = chartKeys?.[0] || 'x';
    const yKey = chartKeys?.[1] || 'y';
    return (
      <ChartWrapper title={title}>
        <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 4 }}>
          <CartesianGrid {...gridDefaults} vertical />
          <XAxis
            dataKey="x"
            name={xKey}
            stroke="none"
            tick={{ fill: DARK.text, fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatNumber}
            label={{ value: xKey, position: 'insideBottom', offset: -2, fill: DARK.text, fontSize: 10 }}
          />
          <YAxis
            dataKey="y"
            name={yKey}
            {...yAxisDefaults}
            label={{ value: yKey, angle: -90, position: 'insideLeft', offset: 10, fill: DARK.text, fontSize: 10 }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const pt = payload[0]?.payload;
              return (
                <div className="custom-tooltip">
                  <p className="label">{xKey}: {formatFullNumber(pt?.x)}</p>
                  <p className="value" style={{ marginTop: 4 }}>{yKey}: {formatFullNumber(pt?.y)}</p>
                </div>
              );
            }}
          />
          <Scatter data={data} fill="#818cf8" opacity={0.6} r={4} />
        </ScatterChart>
      </ChartWrapper>
    );
  }

  // ── Bar Chart ──
  if (chartType === 'bar') {
    return (
      <ChartWrapper title={title}>
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <defs>
            {keys.map((_, i) => (
              <linearGradient key={i} id={`barGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.9} />
                <stop offset="100%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.55} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid {...gridDefaults} />
          <XAxis {...xAxisDefaults(data)} />
          <YAxis {...yAxisDefaults} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(129,140,248,0.04)' }} />
          {keys.length > 1 && <Legend content={<CustomLegend />} />}
          {keys.map((key, i) => (
            <Bar
              key={key}
              dataKey={key}
              fill={`url(#barGrad-${i})`}
              radius={[5, 5, 0, 0]}
              maxBarSize={42}
            />
          ))}
          {showBrush && <Brush {...brushDefaults} />}
        </BarChart>
      </ChartWrapper>
    );
  }

  // ── Grouped Bar ──
  if (chartType === 'grouped_bar') {
    return (
      <ChartWrapper title={title}>
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid {...gridDefaults} />
          <XAxis {...xAxisDefaults(data)} />
          <YAxis {...yAxisDefaults} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(129,140,248,0.04)' }} />
          <Legend content={<CustomLegend />} />
          {keys.map((key, i) => (
            <Bar
              key={key}
              dataKey={key}
              fill={CHART_COLORS[i % CHART_COLORS.length]}
              radius={[4, 4, 0, 0]}
              maxBarSize={32}
            />
          ))}
          {showBrush && <Brush {...brushDefaults} />}
        </BarChart>
      </ChartWrapper>
    );
  }

  // ── Horizontal Bar ──
  if (chartType === 'horizontal_bar') {
    return (
      <ChartWrapper title={title}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 4 }}>
          <CartesianGrid {...gridDefaults} horizontal={false} vertical />
          <XAxis
            type="number"
            stroke="none"
            tick={{ fill: DARK.text, fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatNumber}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="none"
            tick={{ fill: DARK.textBright, fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            width={85}
            tickFormatter={(v) => truncateLabel(v, 12)}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(129,140,248,0.04)' }} />
          {keys.map((key, i) => (
            <Bar
              key={key}
              dataKey={key}
              fill={CHART_COLORS[i % CHART_COLORS.length]}
              radius={[0, 5, 5, 0]}
              maxBarSize={20}
            />
          ))}
        </BarChart>
      </ChartWrapper>
    );
  }

  // ── Line / Area Chart ──
  if (chartType === 'line') {
    return (
      <ChartWrapper title={title}>
        <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <defs>
            {keys.map((key, i) => (
              <linearGradient key={key} id={`areaGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.2} />
                <stop offset="100%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid {...gridDefaults} />
          <XAxis {...xAxisDefaults(data)} />
          <YAxis {...yAxisDefaults} />
          <Tooltip content={<CustomTooltip />} />
          {keys.length > 1 && <Legend content={<CustomLegend />} />}
          {keys.map((key, i) => (
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stroke={CHART_COLORS[i % CHART_COLORS.length]}
              strokeWidth={2}
              fill={`url(#areaGrad-${i})`}
              dot={false}
              activeDot={{
                r: 4,
                fill: CHART_COLORS[i % CHART_COLORS.length],
                stroke: '#131b2e',
                strokeWidth: 2,
              }}
            />
          ))}
          {showBrush && <Brush {...brushDefaults} />}
        </AreaChart>
      </ChartWrapper>
    );
  }

  // ── Pie Chart ──
  if (chartType === 'pie') {
    const valueKey = keys[0] || 'value';
    return (
      <ChartWrapper title={title}>
        <PieChart>
          <defs>
            {data.map((_, i) => (
              <linearGradient key={i} id={`pieGrad-${i}`} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.95} />
                <stop offset="100%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.7} />
              </linearGradient>
            ))}
          </defs>
          <Pie
            data={data}
            dataKey={valueKey}
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={2}
            strokeWidth={0}
            label={renderPieLabel}
            labelLine={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={`url(#pieGrad-${i})`} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend
            content={<CustomLegend />}
            layout="horizontal"
            align="center"
            verticalAlign="bottom"
          />
        </PieChart>
      </ChartWrapper>
    );
  }

  // ── Anomaly Scatter ──
  if (chartType === 'anomaly_scatter') {
    const normal = data.filter(d => !d.is_anomaly);
    const anomalies = data.filter(d => d.is_anomaly);
    return (
      <ChartWrapper title={title || 'Anomaly Detection'}>
        <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid {...gridDefaults} vertical />
          <XAxis
            dataKey="index"
            name="Index"
            stroke="none"
            tick={{ fill: DARK.text, fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatNumber}
          />
          <YAxis
            dataKey="value"
            name="Value"
            {...yAxisDefaults}
          />
          <Tooltip content={<CustomTooltip />} />
          <Scatter name="Normal" data={normal} fill="#818cf8" opacity={0.4} r={3} />
          <Scatter name="Anomaly" data={anomalies} fill="#f87171" opacity={0.85} r={6} shape="diamond" />
          <Legend content={<CustomLegend />} />
        </ScatterChart>
      </ChartWrapper>
    );
  }

  // ── Fallback: Bar ──
  return (
    <ChartWrapper title={title}>
      <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
        <defs>
          {keys.map((_, i) => (
            <linearGradient key={i} id={`fbGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.9} />
              <stop offset="100%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.55} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid {...gridDefaults} />
        <XAxis {...xAxisDefaults(data)} />
        <YAxis {...yAxisDefaults} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(129,140,248,0.04)' }} />
        {keys.length > 1 && <Legend content={<CustomLegend />} />}
        {keys.map((key, i) => (
          <Bar
            key={key}
            dataKey={key}
            fill={`url(#fbGrad-${i})`}
            radius={[5, 5, 0, 0]}
            maxBarSize={42}
          />
        ))}
        {showBrush && <Brush {...brushDefaults} />}
      </BarChart>
    </ChartWrapper>
  );
}

/* ── Box Plot Tooltip ── */
function BoxTooltip({ data: d }) {
  if (!d) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">{d.name}</p>
      <div style={{ marginTop: 4, fontSize: 11, color: DARK.textBright }}>
        <div>Min: {formatFullNumber(d.min)}</div>
        <div>Q1: {formatFullNumber(d.q1)}</div>
        <div>Median: {formatFullNumber(d.median)}</div>
        <div>Q3: {formatFullNumber(d.q3)}</div>
        <div>Max: {formatFullNumber(d.max)}</div>
        {d.mean !== undefined && <div>Mean: {formatFullNumber(d.mean)}</div>}
        {d.outliers?.length > 0 && <div style={{ color: '#f87171', marginTop: 4 }}>Outliers: {d.outliers.length}</div>}
      </div>
    </div>
  );
}
