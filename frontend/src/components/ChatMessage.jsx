import React from 'react';
import { User, Sparkles } from 'lucide-react';
import ChartRenderer from './ChartRenderer';
import DataTable from './DataTable';
import TrustLayer from './TrustLayer';

function formatMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
    .replace(/^\- (.*$)/gm, '<li>$1</li>')
    .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
    .replace(/\n/g, '<br/>');

  html = html.replace(/((?:<li>.*?<\/li><br\/?>?)+)/g, '<ul>$1</ul>');
  html = html.replace(/<br\/><\/ul>/g, '</ul>');
  html = html.replace(/<ul><li>/g, '<ul><li>').replace(/<\/li><br\/><li>/g, '</li><li>');

  return html;
}

export default function ChatMessage({ message, isLast }) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex gap-3 animate-fade-in-up ${isUser ? 'flex-row-reverse' : ''}`}
      style={{ animationDelay: isLast ? '0.08s' : '0s' }}
    >
      {/* Avatar */}
      <div className={`
        shrink-0 w-7 h-7 rounded-lg flex items-center justify-center mt-0.5
        ${isUser
          ? 'bg-surfaceHigh text-textDim'
          : 'bg-primary/12 text-primary'
        }
      `}>
        {isUser ? <User size={13} /> : <Sparkles size={13} />}
      </div>

      {/* Message Body */}
      <div className={`flex flex-col max-w-[82%] min-w-0 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Role Label */}
        <span className={`text-[10px] font-medium mb-1 px-0.5
          ${isUser ? 'text-textDim' : 'text-primary/70'}
        `}>
          {isUser ? 'You' : 'Assistant'}
        </span>

        {/* Text Bubble */}
        <div className={`
          px-4 py-3 text-[13px] leading-relaxed
          ${isUser ? 'chat-bubble-user text-textMain' : 'chat-bubble-ai text-textMuted'}
        `}>
          {message.loading ? (
            <div className="loading-dots py-1">
              <span /><span /><span />
            </div>
          ) : (
            <div
              className="prose-ai"
              dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
            />
          )}
        </div>

        {/* Intent Badge — cleaner, less prominent */}
        {!isUser && message.intent && !message.loading && (
          <span className="mt-1.5 text-[10px] font-medium text-textDim bg-surfaceHigh px-2.5 py-0.5 rounded-full">
            {message.intent.replace('_', ' ')}
          </span>
        )}

        {/* Chart */}
        {!isUser && message.chart_data && (
          <div className="mt-3 w-full chart-container">
            <ChartRenderer
              data={message.chart_data}
              chartType={message.chart_type}
              chartKeys={message.chart_keys}
            />
          </div>
        )}

        {/* Data Table */}
        {!isUser && message.table_data && (
          <div className="mt-3 w-full">
            <DataTable data={message.table_data} />
          </div>
        )}

        {/* Trust Layer */}
        {!isUser && message.trust_layer && !message.loading && (
          <div className="w-full">
            <TrustLayer trustLayer={message.trust_layer} />
          </div>
        )}

        {/* Cached indicator */}
        {message.cached && (
          <span className="mt-1 text-[10px] text-tertiary/70 font-mono">
            ⚡ cached
          </span>
        )}
      </div>
    </div>
  );
}
