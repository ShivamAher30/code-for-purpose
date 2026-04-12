import React from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine, Area, AreaChart, Legend,
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

/* ── Custom Pie Label ── */
function renderPieLabel({ name, percent }) {
  if (percent < 0.03) return null;
  return `${(percent * 100).toFixed(0)}%`;
}

/* ── Legend ── */
const legendStyle = {
  fontSize: 11,
  fontFamily: 'Inter, sans-serif',
  color: DARK.textBright,
  paddingTop: 12,
};

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
function ChartWrapper({ children, title, summary }) {
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
      <div className="w-full h-[320px]">
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

export default function ChartRenderer({ data, chartType, chartKeys, title }) {
  if (!data || data.length === 0) return null;

  const keys = chartKeys || Object.keys(data[0]).filter(
    k => k !== 'name' && k !== 'index' && k !== 'is_anomaly' && k !== 'label'
  );

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
        </AreaChart>
      </ChartWrapper>
    );
  }

  // ── Pie Chart ──
  if (chartType === 'pie') {
    const valueKey = keys[0] || 'value';
    const total = data.reduce((s, d) => s + (Number(d[valueKey]) || 0), 0);
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
      </BarChart>
    </ChartWrapper>
  );
}
