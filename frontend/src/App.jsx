import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, MessageSquare, Activity, Sparkles, ArrowDown,
  BarChart3, Compass, MessageCircle,
} from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';
import DataExplorer from './components/DataExplorer';
import {
  uploadFile, uploadAudio, uploadPDF, uploadImage, uploadText,
  sendQuery, quickAction, clearChat, clearRagData, healthCheck,
  getRagFiles, exportCSV, exportPDF, generateChart,
} from './api';

const WELCOME_SUGGESTIONS = [
  { emoji: '📋', label: 'Summarize my data', query: 'Give me a comprehensive summary of this data' },
  { emoji: '📈', label: 'Show key trends', query: 'Show me the most important trends in this data' },
  { emoji: '🔍', label: 'Find anomalies', query: 'Detect all anomalies and outliers in this data' },
  { emoji: '💡', label: 'Top insights', query: 'What are the top 5 most interesting insights?' },
  { emoji: '📊', label: 'Generate chart', query: 'Create a chart showing the most important metric breakdown' },
  { emoji: '🔄', label: 'Compare segments', query: 'Compare the top segments in this data' },
];

export default function App() {
  // ── State ──
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [schema, setSchema] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [ragResults, setRagResults] = useState([]);
  const [ragFiles, setRagFiles] = useState(null);
  const [routingMode, setRoutingMode] = useState('Auto-Detect');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [activeTab, setActiveTab] = useState('explorer'); // 'explorer' | 'chat'

  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

  // ── Health Check ──
  useEffect(() => {
    const check = async () => {
      const res = await healthCheck();
      setBackendStatus(res.status === 'ok' ? 'online' : 'offline');
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Load RAG files on mount ──
  useEffect(() => {
    const loadRagFiles = async () => {
      try {
        const files = await getRagFiles();
        setRagFiles(files);
      } catch {
        // Backend may not be up yet
      }
    };
    loadRagFiles();
    const interval = setInterval(loadRagFiles, 30000);
    return () => clearInterval(interval);
  }, []);

  // ── Auto-scroll ──
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // ── Scroll detection ──
  const handleScroll = useCallback(() => {
    if (!chatContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
    setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100);
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // ── File Upload (CSV/Excel) ──
  const handleUpload = useCallback(async (file) => {
    setIsUploading(true);
    try {
      const result = await uploadFile(file);
      setUploadResult({ rows: result.rows, columns: result.columns, filename: file.name });
      if (result.schema) setSchema(result.schema);
      setMessages([]);
      setActiveTab('explorer'); // Switch to explorer after upload
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Upload failed: ${err.message}`,
        intent: 'error',
      }]);
    } finally {
      setIsUploading(false);
    }
  }, []);

  // ── RAG File Upload (Audio, PDF, Image, Text) ──
  const handleRagUpload = useCallback(async (file, category) => {
    setIsUploading(true);
    try {
      let result;
      switch (category) {
        case 'audio':
          result = await uploadAudio(file);
          break;
        case 'pdf':
          result = await uploadPDF(file);
          break;
        case 'image':
          result = await uploadImage(file);
          break;
        case 'text':
          result = await uploadText(file);
          break;
        default:
          throw new Error(`Unknown file category: ${category}`);
      }

      setRagResults(prev => [result, ...prev].slice(0, 10));

      // Refresh RAG files list
      try {
        const files = await getRagFiles();
        setRagFiles(files);
      } catch {}

      // Show success message in chat
      const typeLabel = { audio: 'Audio', pdf: 'PDF', image: 'Image', text: 'Text' };
      setActiveTab('chat');
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `**${typeLabel[category] || 'File'} processed successfully.**\n\n**File:** ${result.filename}${result.transcript_preview ? `\n**Preview:** ${result.transcript_preview}` : ''}${result.pages ? `\n**Pages:** ${result.pages}` : ''}${result.chunks_indexed ? `\n**Chunks indexed:** ${result.chunks_indexed}` : ''}${result.images_indexed ? `\n**Images indexed:** ${result.images_indexed}` : ''}\n\nYou can now ask questions about this content.`,
        intent: 'rag',
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Failed to process ${file.name}: ${err.message}`,
        intent: 'error',
      }]);
    } finally {
      setIsUploading(false);
    }
  }, []);

  // ── Send Query ──
  const handleSend = useCallback(async (overrideQuery) => {
    const q = overrideQuery || query.trim();
    if (!q || isProcessing) return;

    // Switch to chat tab when sending a query
    setActiveTab('chat');

    const userMsg = { role: 'user', content: q };
    const loadingMsg = { role: 'assistant', content: '', loading: true };

    setMessages(prev => [...prev, userMsg, loadingMsg]);
    setQuery('');
    setIsProcessing(true);

    try {
      // Detect if user is asking for a chart/graph specifically
      const chartKeywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'draw', 'diagram'];
      const isChartRequest = chartKeywords.some(k => q.toLowerCase().includes(k));

      let result;
      if (isChartRequest && schema) {
        // Use the dedicated chart generation endpoint
        result = await generateChart(q);
        if (result.success) {
          result.intent = 'chart';
          result.trust_layer = { intent: 'chart', pandas_code: result.pandas_code };
        } else {
          // Fall back to regular query
          result = await sendQuery(q, routingMode);
        }
      } else {
        result = await sendQuery(q, routingMode);
      }

      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: result.response,
          intent: result.intent,
          chart_data: result.chart_data,
          chart_type: result.chart_type,
          chart_keys: result.chart_keys,
          table_data: result.table_data,
          trust_layer: result.trust_layer,
          cached: result.cached,
        };
        return updated;
      });
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: `Error: ${err.message}`,
          intent: 'error',
        };
        return updated;
      });
    } finally {
      setIsProcessing(false);
      inputRef.current?.focus();
    }
  }, [query, isProcessing, routingMode, schema]);

  // ── Quick Action ──
  const handleQuickAction = useCallback(async (action) => {
    if (isProcessing) return;

    setActiveTab('chat');
    const labels = { summary: 'Summary', anomaly: 'Anomaly Detection', insights: 'Insights' };
    const userMsg = { role: 'user', content: `Quick Action: ${labels[action] || action}` };
    const loadingMsg = { role: 'assistant', content: '', loading: true };

    setMessages(prev => [...prev, userMsg, loadingMsg]);
    setIsProcessing(true);

    try {
      const result = await quickAction(action);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: result.response,
          intent: result.intent,
          chart_data: result.chart_data,
          chart_type: result.chart_type,
          trust_layer: result.trust_layer,
        };
        return updated;
      });
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: `Error: ${err.message}`,
          intent: 'error',
        };
        return updated;
      });
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  // ── Clear ──
  const handleClear = useCallback(async () => {
    await clearChat();
    setMessages([]);
  }, []);

  // ── Clear RAG Data ──
  const handleClearRag = useCallback(async () => {
    try {
      await clearRagData();
      // Refresh RAG files list to reflect cleared state
      const files = await getRagFiles();
      setRagFiles(files);
      setRagResults([]);
      // Clear all messages for a fully clean slate — backend also wipes
      // its conversation history and query cache on clear-rag.
      setMessages([{
        role: 'assistant',
        content: '**Knowledge base cleared.** All previously uploaded PDFs, audio, images, and text indices have been removed. Conversation history has been reset. You can now upload new files for a fresh context.',
        intent: 'rag',
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Failed to clear RAG data: ${err.message}`,
        intent: 'error',
      }]);
    }
  }, []);

  // ── Export ──
  const handleExportCSV = useCallback(async () => {
    try {
      await exportCSV();
    } catch (err) {
      alert('Export failed: ' + err.message);
    }
  }, []);

  const handleExportPDF = useCallback(async () => {
    try {
      // Send the full chat history (including chart data) to the backend
      const chatHistory = messages
        .filter(m => !m.loading)
        .map(m => ({
          role: m.role,
          content: m.content || '',
          chart_data: m.chart_data || null,
          chart_type: m.chart_type || null,
          chart_keys: m.chart_keys || null,
        }));
      const lastQuery = messages.filter(m => m.role === 'user').pop()?.content || '';
      const lastResponse = messages.filter(m => m.role === 'assistant' && !m.loading).pop()?.content || '';
      await exportPDF(lastQuery, lastResponse, chatHistory);
    } catch (err) {
      alert('PDF export failed: ' + err.message);
    }
  }, [messages]);

  // ── Key handler ──
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasData = !!schema;
  const hasRag = ragFiles && (ragFiles.has_text_index || ragFiles.has_audio_index || ragFiles.has_image_index);
  const chatIsEmpty = messages.length === 0;

  return (
    <div className="flex h-screen overflow-hidden bg-base">
      {/* ── Sidebar ── */}
      <Sidebar
        schema={schema}
        onUpload={handleUpload}
        onRagUpload={handleRagUpload}
        isUploading={isUploading}
        uploadResult={uploadResult}
        ragResults={ragResults}
        ragFiles={ragFiles}
        onQuickAction={handleQuickAction}
        isProcessing={isProcessing}
        routingMode={routingMode}
        setRoutingMode={setRoutingMode}
        onClearChat={handleClear}
        onClearRag={handleClearRag}
        onExportCSV={handleExportCSV}
        onExportPDF={handleExportPDF}
      />

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-surfaceLow">

        {/* Header Bar with Tabs */}
        <header className="h-11 flex items-center justify-between px-6 bg-surfaceContainer/60 backdrop-blur-md shrink-0 z-10">
          <div className="flex items-center gap-3">
            <span className={`
              flex h-1.5 w-1.5 rounded-full
              ${backendStatus === 'online'
                ? 'bg-success'
                : backendStatus === 'offline'
                ? 'bg-error'
                : 'bg-warning animate-pulse'
              }
            `} />
            <span className="text-[10px] font-medium text-textDim">
              {backendStatus === 'online' ? 'Connected' : backendStatus === 'offline' ? 'Offline' : 'Connecting…'}
            </span>
            {uploadResult && (
              <span className="text-[10px] text-textDim/50 tabular-nums">
                · {uploadResult.rows?.toLocaleString()} rows
              </span>
            )}
            {hasRag && (
              <span className="text-[9px] font-medium text-tertiary/70 bg-tertiary/6 px-2 py-0.5 rounded-full">
                RAG
              </span>
            )}
          </div>

          {/* Tab Switcher */}
          {hasData && (
            <div className="tab-switcher">
              <button
                onClick={() => setActiveTab('explorer')}
                className={`tab-btn ${activeTab === 'explorer' ? 'active' : ''}`}
              >
                <Compass size={13} />
                <span>Explorer</span>
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
              >
                <MessageCircle size={13} />
                <span>Chat</span>
                {messages.length > 0 && (
                  <span className="tab-badge">{messages.length}</span>
                )}
              </button>
            </div>
          )}

          <div className="flex items-center gap-3">
            {uploadResult && (
              <span className="text-[10px] text-textDim/50 bg-surfaceHigh/60 px-2.5 py-1 rounded-lg">
                {uploadResult.filename}
              </span>
            )}
          </div>
        </header>

        {/* ── Explorer View ── */}
        {hasData && activeTab === 'explorer' && (
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <DataExplorer schema={schema} uploadResult={uploadResult} />
          </div>
        )}

        {/* ── Chat View ── */}
        {(activeTab === 'chat' || !hasData) && (
          <>
            <div
              ref={chatContainerRef}
              onScroll={handleScroll}
              className="flex-1 overflow-y-auto px-6 py-6 pb-40"
            >
              {chatIsEmpty ? (
                /* ── Welcome State ── */
                <div className="flex flex-col items-center justify-center h-full max-w-xl mx-auto text-center animate-fade-in">
                  <div className="welcome-orb mb-6 animate-orb-float">
                    <Sparkles size={28} className="text-primary" />
                  </div>
                  <h2 className="font-display text-3xl font-bold text-textMain mb-3 leading-tight tracking-tight">
                    Talk-to-Data <span className="gradient-text">AI</span>
                  </h2>
                  <p className="text-sm text-textMuted mb-10 leading-relaxed max-w-md">
                    {hasData
                      ? 'Your data is ready. Ask anything — from simple queries to complex analysis, chart generation, comparisons, and anomaly detection.'
                      : hasRag
                      ? 'Knowledge base loaded. Ask questions about your uploaded documents, audio, and images.'
                      : 'Upload a CSV, Excel, PDF, audio, or image file from the sidebar to get started.'}
                  </p>

                  {(hasData || hasRag) && (
                    <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
                      {WELCOME_SUGGESTIONS.map(({ emoji, label, query: q }) => (
                        <button
                          key={q}
                          onClick={() => handleSend(q)}
                          disabled={isProcessing}
                          className="suggestion-chip"
                        >
                          <span className="text-base mr-1">{emoji}</span> {label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                /* ── Messages ── */
                <div className="max-w-4xl mx-auto space-y-6">
                  {messages.map((msg, i) => (
                    <ChatMessage
                      key={i}
                      message={msg}
                      isLast={i === messages.length - 1}
                    />
                  ))}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>

            {/* Scroll to Bottom Button */}
            {showScrollBtn && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-40 right-8 z-20 p-2 rounded-xl bg-surfaceHigh/80 backdrop-blur-sm
                  text-textDim hover:text-textMain hover:bg-surfaceBright/80
                  transition-all shadow-lg"
                style={{ boxShadow: '0 8px 24px -8px rgba(6,14,32,0.5)' }}
              >
                <ArrowDown size={14} />
              </button>
            )}

            {/* ── Floating Input ── */}
            <div className="absolute bottom-0 left-0 right-0 p-5 pt-20 bg-gradient-to-t from-surfaceLow via-surfaceLow/95 to-transparent pointer-events-none z-10">
              <div className="max-w-4xl mx-auto pointer-events-auto">
                <div className="input-bar p-1.5 flex items-end gap-2">
                  <div className="pl-3 pb-2.5 flex items-center gap-1">
                    <MessageSquare size={14} className="text-textDim/30" />
                  </div>
                  <textarea
                    ref={inputRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                      hasData
                        ? "Ask anything about your data…"
                        : hasRag
                        ? "Ask about your documents…"
                        : "Upload data to get started…"
                    }
                    disabled={backendStatus === 'offline'}
                    rows={1}
                    className="
                      flex-1 bg-transparent border-0 outline-none resize-none
                      text-textMain placeholder:text-textDim/30
                      text-[13px] leading-relaxed py-2.5
                      max-h-32 min-h-[36px]
                      disabled:opacity-50
                    "
                    style={{ fieldSizing: 'content' }}
                  />
                  <div className="flex items-center gap-1.5 pb-1">
                    <button
                      onClick={() => handleSend()}
                      disabled={!query.trim() || isProcessing || backendStatus === 'offline'}
                      className={`
                        p-2.5 rounded-xl transition-all duration-200 shrink-0
                        ${query.trim() && !isProcessing
                          ? 'bg-primary text-onPrimary hover:bg-primaryContainer'
                          : 'bg-surfaceHigh/50 text-textDim/30 cursor-not-allowed'
                        }
                      `}
                    >
                      {isProcessing ? (
                        <div className="w-[14px] h-[14px] border-2 border-white/20 border-t-white rounded-full" style={{ animation: 'spin 0.7s linear infinite' }} />
                      ) : (
                        <Send size={14} strokeWidth={2.5} />
                      )}
                    </button>
                  </div>
                </div>
                <p className="text-center text-[9px] text-textDim/25 mt-2 font-medium">
                  AI-powered analysis · Charts, summaries, anomalies, comparisons
                </p>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
