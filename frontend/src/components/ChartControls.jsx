import React, { useState } from 'react';
import {
  BarChart3, LineChart, PieChart, ArrowUpDown, Filter, Maximize2, Minimize2,
  TrendingUp, ScatterChart, LayoutGrid as Grid,
} from 'lucide-react';

const CHART_TYPE_OPTIONS = [
  { value: 'bar', icon: BarChart3, label: 'Bar' },
  { value: 'line', icon: LineChart, label: 'Line' },
  { value: 'pie', icon: PieChart, label: 'Pie' },
  { value: 'scatter', icon: ScatterChart, label: 'Scatter' },
  { value: 'histogram', icon: BarChart3, label: 'Histogram' },
];

const AGG_OPTIONS = [
  { value: 'sum', label: 'Sum' },
  { value: 'mean', label: 'Average' },
  { value: 'count', label: 'Count' },
  { value: 'min', label: 'Min' },
  { value: 'max', label: 'Max' },
];

const SORT_OPTIONS = [
  { value: null, label: 'Default' },
  { value: 'value_desc', label: 'Value ↓' },
  { value: 'value_asc', label: 'Value ↑' },
  { value: 'label_asc', label: 'Name A-Z' },
  { value: 'label_desc', label: 'Name Z-A' },
];

export default function ChartControls({
  suggestion,
  onChartTypeChange,
  onAggregationChange,
  onSortChange,
  onFilterChange,
  onFullscreen,
  isFullscreen,
  filterOptions,
  currentAggregation,
  currentSort,
  currentChartType,
  activeFilters,
}) {
  const [showFilters, setShowFilters] = useState(false);

  const compatibleTypes = getCompatibleTypes(suggestion?.chart_type);

  return (
    <div className="chart-controls">
      {/* Chart Type Switcher */}
      <div className="chart-control-group">
        <span className="chart-control-label">Type</span>
        <div className="chart-type-pills">
          {compatibleTypes.map(({ value, icon: Icon, label }) => (
            <button
              key={value}
              onClick={() => onChartTypeChange(value)}
              className={`chart-type-pill ${currentChartType === value ? 'active' : ''}`}
              title={label}
            >
              <Icon size={13} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Aggregation */}
      {suggestion?.chart_type !== 'histogram' && suggestion?.chart_type !== 'heatmap' && (
        <div className="chart-control-group">
          <span className="chart-control-label">Aggregate</span>
          <select
            value={currentAggregation || 'sum'}
            onChange={(e) => onAggregationChange(e.target.value)}
            className="chart-control-select"
          >
            {AGG_OPTIONS.map(({ value, label }) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Sort */}
      {suggestion?.chart_type !== 'heatmap' && suggestion?.chart_type !== 'scatter' && (
        <div className="chart-control-group">
          <span className="chart-control-label">Sort</span>
          <select
            value={currentSort || ''}
            onChange={(e) => onSortChange(e.target.value || null)}
            className="chart-control-select"
          >
            {SORT_OPTIONS.map(({ value, label }) => (
              <option key={label} value={value || ''}>{label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Filter Toggle */}
      {filterOptions && Object.keys(filterOptions).length > 0 && (
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`chart-control-btn ${showFilters ? 'active' : ''}`}
        >
          <Filter size={13} />
          <span>Filters</span>
          {activeFilters && Object.keys(activeFilters).length > 0 && (
            <span className="filter-badge">{Object.keys(activeFilters).length}</span>
          )}
        </button>
      )}

      {/* Fullscreen */}
      <button
        onClick={onFullscreen}
        className="chart-control-btn ml-auto"
        title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
      >
        {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
      </button>

      {/* Filter Panel */}
      {showFilters && filterOptions && (
        <div className="chart-filter-panel">
          {Object.entries(filterOptions).map(([col, values]) => (
            <div key={col} className="chart-filter-group">
              <label className="chart-filter-label">{col}</label>
              <div className="chart-filter-chips">
                {values.slice(0, 15).map((val) => {
                  const isActive = activeFilters?.[col]?.includes(val);
                  return (
                    <button
                      key={val}
                      onClick={() => {
                        const current = activeFilters?.[col] || [];
                        const updated = isActive
                          ? current.filter((v) => v !== val)
                          : [...current, val];
                        onFilterChange(col, updated.length > 0 ? updated : null);
                      }}
                      className={`filter-chip ${isActive ? 'active' : ''}`}
                    >
                      {val}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function getCompatibleTypes(baseType) {
  const compatMap = {
    bar: ['bar', 'line', 'pie'],
    grouped_bar: ['bar', 'line'],
    line: ['line', 'bar'],
    pie: ['pie', 'bar'],
    histogram: ['histogram'],
    box: ['box'],
    scatter: ['scatter'],
    heatmap: ['heatmap'],
  };
  const compatible = compatMap[baseType] || ['bar', 'line', 'pie'];
  return CHART_TYPE_OPTIONS.filter((o) => compatible.includes(o.value));
}
