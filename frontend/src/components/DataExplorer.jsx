import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart3, LineChart, PieChart, ScatterChart, Activity,
  TrendingUp, Layers, GitBranch, Sparkles, Loader2,
  LayoutGrid, ChevronDown, ChevronRight, Target, Boxes,
} from 'lucide-react';
import { getDatasetProfile, renderSuggestionChart } from '../api';
import ChartRenderer from './ChartRenderer';
import ChartControls from './ChartControls';
import InsightCards from './InsightCards';

const CATEGORY_ICONS = {
  Trends: TrendingUp,
  Comparisons: BarChart3,
  Distributions: Activity,
  Relationships: GitBranch,
  Compositions: PieChart,
};

const CATEGORY_COLORS = {
  Trends: '#60a5fa',
  Comparisons: '#818cf8',
  Distributions: '#f59e0b',
  Relationships: '#a78bfa',
  Compositions: '#34d399',
};

const CHART_TYPE_ICONS = {
  bar: BarChart3,
  grouped_bar: BarChart3,
  line: LineChart,
  pie: PieChart,
  scatter: ScatterChart,
  histogram: Activity,
  box: Target,
  heatmap: LayoutGrid,
};

export default function DataExplorer({ schema, uploadResult }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeChart, setActiveChart] = useState(null);
  const [activeChartData, setActiveChartData] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState({});
  const [customization, setCustomization] = useState({
    aggregation: 'sum',
    sort_by: null,
    filters: {},
    chart_type_override: null,
  });

  // Fetch profile on mount
  useEffect(() => {
    let cancelled = false;
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getDatasetProfile();
        if (!cancelled) {
          setProfile(data);
          // Auto-expand first two categories
          const cats = [...new Set(data.suggestions.map((s) => s.category))];
          const expanded = {};
          cats.slice(0, 2).forEach((c) => (expanded[c] = true));
          setExpandedCategories(expanded);

          // Auto-select first pre-rendered chart
          if (data.pre_rendered && Object.keys(data.pre_rendered).length > 0) {
            const firstKey = Object.keys(data.pre_rendered)[0];
            const firstSuggestion = data.suggestions.find((s) => s.id === firstKey);
            if (firstSuggestion) {
              setActiveChart(firstSuggestion);
              setActiveChartData(data.pre_rendered[firstKey]);
            }
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchProfile();
    return () => { cancelled = true; };
  }, [schema]);

  // Render a specific chart
  const handleSelectSuggestion = useCallback(async (suggestion) => {
    setActiveChart(suggestion);
    setChartLoading(true);
    setCustomization({
      aggregation: 'sum',
      sort_by: null,
      filters: {},
      chart_type_override: null,
    });

    // Check pre-rendered cache
    if (profile?.pre_rendered?.[suggestion.id]) {
      setActiveChartData(profile.pre_rendered[suggestion.id]);
      setChartLoading(false);
      return;
    }

    try {
      const result = await renderSuggestionChart(suggestion.id);
      setActiveChartData(result);
    } catch (err) {
      setActiveChartData(null);
    } finally {
      setChartLoading(false);
    }
  }, [profile]);

  // Re-render with customization
  const handleCustomizationChange = useCallback(async (updates) => {
    const newCustomization = { ...customization, ...updates };
    setCustomization(newCustomization);
    if (!activeChart) return;

    setChartLoading(true);
    try {
      const result = await renderSuggestionChart(activeChart.id, {
        aggregation: newCustomization.aggregation,
        sort_by: newCustomization.sort_by,
        filters: newCustomization.filters,
        chart_type_override: newCustomization.chart_type_override,
      });
      setActiveChartData(result);
    } catch (err) {
      // Keep current chart
    } finally {
      setChartLoading(false);
    }
  }, [customization, activeChart]);

  const toggleCategory = (cat) => {
    setExpandedCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  if (loading) {
    return (
      <div className="explorer-loading">
        <div className="explorer-loading-spinner">
          <Loader2 size={24} className="animate-spin" style={{ color: '#818cf8' }} />
        </div>
        <p>Analyzing your dataset…</p>
        <span>Detecting patterns, correlations, and optimal visualizations</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="explorer-loading">
        <p style={{ color: '#f87171' }}>Failed to profile dataset: {error}</p>
      </div>
    );
  }

  if (!profile) return null;

  const { suggestions, insights, summary, quality_score, filter_options } = profile;
  const groupedSuggestions = {};
  suggestions.forEach((s) => {
    if (!groupedSuggestions[s.category]) groupedSuggestions[s.category] = [];
    groupedSuggestions[s.category].push(s);
  });

  return (
    <div className={`data-explorer ${isFullscreen ? 'fullscreen' : ''}`}>
      {/* ── Dataset Overview ── */}
      <div className="explorer-header animate-fade-in">
        <div className="explorer-title-row">
          <div className="explorer-title">
            <Sparkles size={18} className="text-primary" />
            <div>
              <h2>Dataset Explorer</h2>
              <p>{uploadResult?.filename || 'Uploaded Dataset'}</p>
            </div>
          </div>
          <div className="quality-score" title="Data Quality Score">
            <svg viewBox="0 0 36 36" className="quality-ring">
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="rgba(70,69,84,0.2)"
                strokeWidth="3"
              />
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke={quality_score >= 80 ? '#34d399' : quality_score >= 50 ? '#fbbf24' : '#f87171'}
                strokeWidth="3"
                strokeDasharray={`${quality_score}, 100`}
                strokeLinecap="round"
              />
            </svg>
            <span className="quality-value">{Math.round(quality_score)}</span>
          </div>
        </div>

        <div className="explorer-stats">
          <div className="explorer-stat">
            <span className="stat-value">{summary.rows?.toLocaleString()}</span>
            <span className="stat-label">Rows</span>
          </div>
          <div className="explorer-stat">
            <span className="stat-value">{summary.columns}</span>
            <span className="stat-label">Columns</span>
          </div>
          <div className="explorer-stat">
            <span className="stat-value">{summary.numeric_count}</span>
            <span className="stat-label">Numeric</span>
          </div>
          <div className="explorer-stat">
            <span className="stat-value">{summary.categorical_count}</span>
            <span className="stat-label">Categorical</span>
          </div>
          {summary.date_count > 0 && (
            <div className="explorer-stat">
              <span className="stat-value">{summary.date_count}</span>
              <span className="stat-label">Date</span>
            </div>
          )}
          {summary.missing_pct > 0 && (
            <div className="explorer-stat warn">
              <span className="stat-value">{summary.missing_pct}%</span>
              <span className="stat-label">Missing</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Auto Insights ── */}
      {insights.length > 0 && (
        <div className="explorer-section animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <h3 className="explorer-section-title">
            <Sparkles size={14} /> Key Insights
            <span className="insight-count">{insights.length}</span>
          </h3>
          <InsightCards insights={insights} />
        </div>
      )}

      {/* ── Smart Chart Suggestions ── */}
      <div className="explorer-section animate-fade-in" style={{ animationDelay: '0.2s' }}>
        <h3 className="explorer-section-title">
          <Boxes size={14} /> Recommended Visualizations
          <span className="insight-count">{suggestions.length}</span>
        </h3>

        <div className="suggestion-categories">
          {Object.entries(groupedSuggestions).map(([category, items]) => {
            const CatIcon = CATEGORY_ICONS[category] || BarChart3;
            const catColor = CATEGORY_COLORS[category] || '#818cf8';
            const isExpanded = expandedCategories[category];

            return (
              <div key={category} className="suggestion-category">
                <button
                  className="category-header"
                  onClick={() => toggleCategory(category)}
                >
                  <div className="category-header-left">
                    <CatIcon size={14} style={{ color: catColor }} />
                    <span>{category}</span>
                    <span className="category-count" style={{ color: catColor }}>{items.length}</span>
                  </div>
                  {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </button>

                {isExpanded && (
                  <div className="suggestion-grid animate-fade-in">
                    {items.map((s) => {
                      const Icon = CHART_TYPE_ICONS[s.chart_type] || BarChart3;
                      const isActive = activeChart?.id === s.id;

                      return (
                        <button
                          key={s.id}
                          onClick={() => handleSelectSuggestion(s)}
                          className={`suggestion-card ${isActive ? 'active' : ''}`}
                        >
                          <div className="suggestion-card-header">
                            <Icon size={15} style={{ color: catColor }} />
                            <span className="suggestion-chart-type">{s.chart_type.replace('_', ' ')}</span>
                          </div>
                          <p className="suggestion-label">{s.label}</p>
                          <p className="suggestion-reason">{s.reason}</p>
                          <div className="suggestion-cols">
                            {s.columns.map((c) => (
                              <span key={c} className="col-tag">{c}</span>
                            ))}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Active Chart ── */}
      {activeChart && (
        <div className={`explorer-chart-area animate-fade-in ${isFullscreen ? 'chart-fullscreen' : ''}`}>
          <ChartControls
            suggestion={activeChart}
            currentChartType={customization.chart_type_override || activeChart.chart_type}
            currentAggregation={customization.aggregation}
            currentSort={customization.sort_by}
            activeFilters={customization.filters}
            filterOptions={filter_options}
            isFullscreen={isFullscreen}
            onChartTypeChange={(type) =>
              handleCustomizationChange({ chart_type_override: type })
            }
            onAggregationChange={(agg) =>
              handleCustomizationChange({ aggregation: agg })
            }
            onSortChange={(sort) =>
              handleCustomizationChange({ sort_by: sort })
            }
            onFilterChange={(col, vals) => {
              const newFilters = { ...customization.filters };
              if (vals) {
                newFilters[col] = vals;
              } else {
                delete newFilters[col];
              }
              handleCustomizationChange({ filters: newFilters });
            }}
            onFullscreen={() => setIsFullscreen(!isFullscreen)}
          />

          <div className="explorer-chart-content">
            {chartLoading ? (
              <div className="explorer-chart-loading">
                <Loader2 size={20} className="animate-spin" style={{ color: '#818cf8' }} />
                <span>Rendering chart…</span>
              </div>
            ) : activeChartData ? (
              <ChartRenderer
                data={activeChartData.chart_data}
                chartType={activeChartData.chart_type}
                chartKeys={activeChartData.chart_keys}
                title={activeChart.label}
              />
            ) : (
              <p className="text-textDim text-sm text-center py-12">
                Could not render this chart. Try a different suggestion.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
