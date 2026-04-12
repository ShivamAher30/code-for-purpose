import React from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, Link, BarChart3, Target, RefreshCw } from 'lucide-react';

const SEVERITY_STYLES = {
  warning: { bg: 'rgba(251, 191, 36, 0.08)', border: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' },
  info: { bg: 'rgba(129, 140, 248, 0.06)', border: 'rgba(129, 140, 248, 0.15)', color: '#818cf8' },
  error: { bg: 'rgba(248, 113, 113, 0.08)', border: 'rgba(248, 113, 113, 0.2)', color: '#f87171' },
};

const TYPE_ICONS = {
  data_quality: AlertTriangle,
  distribution: BarChart3,
  anomaly: Target,
  correlation: Link,
  trend: TrendingUp,
};

export default function InsightCards({ insights, onInsightClick }) {
  if (!insights || insights.length === 0) return null;

  return (
    <div className="insight-cards-grid">
      {insights.map((insight, i) => {
        const style = SEVERITY_STYLES[insight.severity] || SEVERITY_STYLES.info;
        const Icon = TYPE_ICONS[insight.type] || BarChart3;

        return (
          <button
            key={i}
            className="insight-card"
            style={{
              background: style.bg,
              borderColor: style.border,
            }}
            onClick={() => onInsightClick?.(insight)}
          >
            <div className="insight-card-icon" style={{ color: style.color }}>
              <span className="text-base mr-1">{insight.icon}</span>
            </div>
            <p className="insight-card-text">{insight.text}</p>
            <span
              className="insight-type-badge"
              style={{ color: style.color, background: `${style.color}15` }}
            >
              {insight.type.replace('_', ' ')}
            </span>
          </button>
        );
      })}
    </div>
  );
}
