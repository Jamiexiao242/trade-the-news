import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, AlertTriangle, DollarSign, Clock, Activity, Terminal, Zap } from 'lucide-react';

const MetricsAnalyzer = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTimeline, setSelectedTimeline] = useState(null);
  const [timelineLimit, setTimelineLimit] = useState(30);
  const API_BASE = import.meta?.env?.VITE_METRICS_API || 'http://localhost:2010';

  const fetchRemote = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/metrics?limit=500`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      analyzeMetrics(data);
    } catch (e) {
      setError(`ERROR: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const analyzeMetrics = (data) => {
    // Process in chronological order
    const entries = [...data].sort((a, b) => (a.ts || 0) - (b.ts || 0));
    const byKind = {};
    const ordersBySymbol = {};
    const timelines = [];
    let currentTimeline = null;

    const performance = {
      llm_latencies: { filter: [], direction: [], risk: [] },
      processing_times: [],
      conversion_funnel: {
        news_in: 0,
        passed_filter: 0,
        got_direction: 0,
        passed_risk: 0,
        order_submitted: 0,
        dropped_holdings: 0,
      },
    };

    entries.forEach((entry) => {
      byKind[entry.kind] = (byKind[entry.kind] || 0) + 1;

      if (entry.kind === 'news_in') performance.conversion_funnel.news_in++;
      if (entry.kind === 'llm_filter' && entry.payload?.result?.actionable) performance.conversion_funnel.passed_filter++;
      if (entry.kind === 'llm_direction') performance.conversion_funnel.got_direction++;
      if (entry.kind === 'llm_risk' && entry.payload?.result?.approve) performance.conversion_funnel.passed_risk++;
      if (entry.kind === 'order_submitted') performance.conversion_funnel.order_submitted++;
      if (entry.kind === 'dropped_holdings') performance.conversion_funnel.dropped_holdings++;

      if (entry.kind === 'order_submitted') {
        const symbol = entry.payload?.symbol;
        if (symbol) {
          if (!ordersBySymbol[symbol]) {
            ordersBySymbol[symbol] = {
              count: 0,
              total_qty: 0,
              actions: {},
              avg_price: 0,
              total_price: 0,
              avg_atr: 0,
              total_atr: 0,
              trail_params: [],
            };
          }
          ordersBySymbol[symbol].count++;
          ordersBySymbol[symbol].total_qty += entry.payload.qty || 0;
          ordersBySymbol[symbol].total_price += entry.payload.last_price || 0;
          ordersBySymbol[symbol].total_atr += entry.payload.atr || 0;
          ordersBySymbol[symbol].trail_params.push({
            wide: entry.payload.wide_trail,
            tight: entry.payload.tight_trail,
            breakeven: entry.payload.breakeven_pct,
          });
          const action = entry.payload.action;
          ordersBySymbol[symbol].actions[action] = (ordersBySymbol[symbol].actions[action] || 0) + 1;
        }
      }

      if (entry.kind === 'news_in') {
        if (currentTimeline) {
          const start = currentTimeline.events[0].ts;
          const end = currentTimeline.events[currentTimeline.events.length - 1].ts;
          currentTimeline.processing_time = (end - start) * 1000;
          performance.processing_times.push(currentTimeline.processing_time);
          timelines.push(currentTimeline);
        }
        currentTimeline = {
          ts: entry.ts,
          headline: entry.payload?.headline || 'N/A',
          tickers: entry.payload?.tickers || [],
          events: [entry],
          processing_time: 0,
        };
      } else if (currentTimeline) {
        currentTimeline.events.push(entry);
      }
    });

    if (currentTimeline) {
      const start = currentTimeline.events[0].ts;
      const end = currentTimeline.events[currentTimeline.events.length - 1].ts;
      currentTimeline.processing_time = (end - start) * 1000;
      performance.processing_times.push(currentTimeline.processing_time);
      timelines.push(currentTimeline);
    }

    timelines.forEach((tl) => {
      const getEvent = (k) => tl.events.find((e) => e.kind === k);
      const news = tl.events[0];
      const f = getEvent('llm_filter');
      const d = getEvent('llm_direction');
      const r = getEvent('llm_risk');
      if (news && f) performance.llm_latencies.filter.push((f.ts - news.ts) * 1000);
      if (f && d) performance.llm_latencies.direction.push((d.ts - f.ts) * 1000);
      if (d && r) performance.llm_latencies.risk.push((r.ts - d.ts) * 1000);
    });

    Object.keys(ordersBySymbol).forEach((symbol) => {
      const data = ordersBySymbol[symbol];
      data.avg_price = data.total_price / data.count;
      data.avg_atr = data.total_atr / data.count;
    });

    const avgLatency = (arr) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    performance.avg_latencies = {
      filter: avgLatency(performance.llm_latencies.filter),
      direction: avgLatency(performance.llm_latencies.direction),
      risk: avgLatency(performance.llm_latencies.risk),
      total: avgLatency([
        ...performance.llm_latencies.filter,
        ...performance.llm_latencies.direction,
        ...performance.llm_latencies.risk,
      ]),
    };

    performance.avg_processing_time = avgLatency(performance.processing_times);

    const funnel = performance.conversion_funnel;
    performance.conversion_rates = {
      filter_pass: funnel.news_in > 0 ? (funnel.passed_filter / funnel.news_in) * 100 : 0,
      risk_pass: funnel.got_direction > 0 ? (funnel.passed_risk / funnel.got_direction) * 100 : 0,
      final_conversion: funnel.news_in > 0 ? (funnel.order_submitted / funnel.news_in) * 100 : 0,
      drop_rate: funnel.news_in > 0 ? (funnel.dropped_holdings / funnel.news_in) * 100 : 0,
    };

    setStats({
      byKind,
      ordersBySymbol,
      timelines,
      performance,
      total: data.length,
    });
  };

  const formatTimestamp = (ts) => {
    return new Date(ts * 1000).toLocaleString('en-US', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  useEffect(() => {
    fetchRemote();
    const id = setInterval(fetchRemote, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen bg-black text-orange-500 p-4 font-mono">
      <div className="max-w-[1800px] mx-auto">
        {/* Header */}
        <div className="border-2 border-orange-500 bg-black p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Terminal className="w-8 h-8 text-orange-500" />
              <div>
                <h1 className="text-2xl font-bold tracking-wider">METRICS ANALYZER TERMINAL</h1>
                <p className="text-xs text-orange-400">REAL-TIME TRADING INTELLIGENCE SYSTEM</p>
              </div>
            </div>
            <div className="text-right text-sm">
              <div className="text-orange-400">{new Date().toLocaleDateString('en-US')}</div>
              <div className="text-cyan-400">{new Date().toLocaleTimeString('en-US', { hour12: false })}</div>
            </div>
          </div>
        </div>

        {/* Live Fetch */}
        <div className="border-2 border-orange-500 bg-black p-6 mb-4 flex items-center justify-between">
          <div>
            <div className="text-orange-400 text-sm">LIVE METRICS</div>
            <div className="text-xs text-orange-600">Polling {API_BASE}/metrics?limit=500 every 3s</div>
          </div>
          <button
            onClick={fetchRemote}
            className="px-3 py-2 border border-orange-500 text-orange-400 hover:text-cyan-400 hover:border-cyan-400 text-xs"
          >
            REFRESH NOW
          </button>
          {loading && <p className="text-cyan-400 animate-pulse text-xs">LOADING...</p>}
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>

        {stats && (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-4 mb-4">
              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-xs text-orange-400 mb-1">TOTAL EVENTS</div>
                <div className="text-3xl font-bold text-cyan-400">{stats.total.toLocaleString()}</div>
                <div className="text-xs text-orange-600 mt-1">RECORDS PROCESSED</div>
              </div>

              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-xs text-orange-400 mb-1">NEWS FEED</div>
                <div className="text-3xl font-bold text-green-400">{(stats.byKind.news_in || 0).toLocaleString()}</div>
                <div className="text-xs text-orange-600 mt-1">HEADLINES INGESTED</div>
              </div>

              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-xs text-orange-400 mb-1">ORDERS EXEC</div>
                <div className="text-3xl font-bold text-yellow-400">{(stats.byKind.order_submitted || 0).toLocaleString()}</div>
                <div className="text-xs text-orange-600 mt-1">TRADES SUBMITTED</div>
              </div>

              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-xs text-orange-400 mb-1">SYMBOLS</div>
                <div className="text-3xl font-bold text-purple-400">{Object.keys(stats.ordersBySymbol).length}</div>
                <div className="text-xs text-orange-600 mt-1">UNIQUE TICKERS</div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="border-2 border-orange-500 bg-black p-4 mb-4">
              <div className="text-sm text-orange-400 mb-3 border-b border-orange-500 pb-2 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                PERFORMANCE METRICS
              </div>
              
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div className="border border-orange-500 bg-orange-950 bg-opacity-20 p-3">
                  <div className="text-xs text-orange-400 mb-1">AVG PROCESSING TIME</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {stats.performance.avg_processing_time.toFixed(0)}ms
                  </div>
                  <div className="text-xs text-orange-600 mt-1">NEWS → ORDER</div>
                </div>

                <div className="border border-orange-500 bg-orange-950 bg-opacity-20 p-3">
                  <div className="text-xs text-orange-400 mb-1">FILTER PASS RATE</div>
                  <div className="text-2xl font-bold text-green-400">
                    {stats.performance.conversion_rates.filter_pass.toFixed(1)}%
                  </div>
                  <div className="text-xs text-orange-600 mt-1">
                    {stats.performance.conversion_funnel.passed_filter}/{stats.performance.conversion_funnel.news_in}
                  </div>
                </div>

                <div className="border border-orange-500 bg-orange-950 bg-opacity-20 p-3">
                  <div className="text-xs text-orange-400 mb-1">RISK APPROVAL RATE</div>
                  <div className="text-2xl font-bold text-yellow-400">
                    {stats.performance.conversion_rates.risk_pass.toFixed(1)}%
                  </div>
                  <div className="text-xs text-orange-600 mt-1">
                    {stats.performance.conversion_funnel.passed_risk}/{stats.performance.conversion_funnel.got_direction}
                  </div>
                </div>

                <div className="border border-orange-500 bg-orange-950 bg-opacity-20 p-3">
                  <div className="text-xs text-orange-400 mb-1">FINAL CONVERSION</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {stats.performance.conversion_rates.final_conversion.toFixed(1)}%
                  </div>
                  <div className="text-xs text-orange-600 mt-1">NEWS → ORDER</div>
                </div>
              </div>

              {(stats.performance.llm_latencies.filter.length > 0 || 
                stats.performance.llm_latencies.direction.length > 0 || 
                stats.performance.llm_latencies.risk.length > 0) && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="border border-cyan-500 bg-cyan-950 bg-opacity-20 p-3">
                    <div className="text-xs text-cyan-400 mb-1">LLM FILTER LATENCY</div>
                    <div className="text-xl font-bold text-cyan-400">
                      {stats.performance.avg_latencies.filter.toFixed(0)}ms
                    </div>
                    <div className="text-xs text-cyan-600 mt-1">
                      {stats.performance.llm_latencies.filter.length} calls
                    </div>
                  </div>

                  <div className="border border-cyan-500 bg-cyan-950 bg-opacity-20 p-3">
                    <div className="text-xs text-cyan-400 mb-1">LLM DIRECTION LATENCY</div>
                    <div className="text-xl font-bold text-cyan-400">
                      {stats.performance.avg_latencies.direction.toFixed(0)}ms
                    </div>
                    <div className="text-xs text-cyan-600 mt-1">
                      {stats.performance.llm_latencies.direction.length} calls
                    </div>
                  </div>

                  <div className="border border-cyan-500 bg-cyan-950 bg-opacity-20 p-3">
                    <div className="text-xs text-cyan-400 mb-1">LLM RISK LATENCY</div>
                    <div className="text-xl font-bold text-cyan-400">
                      {stats.performance.avg_latencies.risk.toFixed(0)}ms
                    </div>
                    <div className="text-xs text-cyan-600 mt-1">
                      {stats.performance.llm_latencies.risk.length} calls
                    </div>
                  </div>

                  <div className="border border-cyan-500 bg-cyan-950 bg-opacity-20 p-3">
                    <div className="text-xs text-cyan-400 mb-1">TOTAL LLM LATENCY</div>
                    <div className="text-xl font-bold text-cyan-400">
                      {stats.performance.avg_latencies.total.toFixed(0)}ms
                    </div>
                    <div className="text-xs text-cyan-600 mt-1">AVERAGE</div>
                  </div>
                </div>
              )}

              <div className="mt-4 border-t border-orange-500 pt-4">
                <div className="text-xs text-orange-400 mb-3">CONVERSION FUNNEL</div>
                <div className="flex items-center justify-between text-xs">
                  <div className="text-center flex-1">
                    <div className="text-cyan-400 font-bold text-lg">{stats.performance.conversion_funnel.news_in}</div>
                    <div className="text-orange-400">NEWS IN</div>
                  </div>
                  <div className="text-orange-600">→</div>
                  <div className="text-center flex-1">
                    <div className="text-green-400 font-bold text-lg">{stats.performance.conversion_funnel.passed_filter}</div>
                    <div className="text-orange-400">FILTERED</div>
                  </div>
                  <div className="text-orange-600">→</div>
                  <div className="text-center flex-1">
                    <div className="text-blue-400 font-bold text-lg">{stats.performance.conversion_funnel.got_direction}</div>
                    <div className="text-orange-400">DIRECTION</div>
                  </div>
                  <div className="text-orange-600">→</div>
                  <div className="text-center flex-1">
                    <div className="text-yellow-400 font-bold text-lg">{stats.performance.conversion_funnel.passed_risk}</div>
                    <div className="text-orange-400">RISK OK</div>
                  </div>
                  <div className="text-orange-600">→</div>
                  <div className="text-center flex-1">
                    <div className="text-purple-400 font-bold text-lg">{stats.performance.conversion_funnel.order_submitted}</div>
                    <div className="text-orange-400">ORDERS</div>
                  </div>
                </div>
                {stats.performance.conversion_funnel.dropped_holdings > 0 && (
                  <div className="mt-2 text-center text-red-400 text-xs">
                    {stats.performance.conversion_funnel.dropped_holdings} DROPPED (ALREADY HOLDING)
                  </div>
                )}
              </div>
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-sm text-orange-400 mb-3 border-b border-orange-500 pb-2">
                  EVENT TYPE DISTRIBUTION
                </div>
                <div className="overflow-auto">
                  <BarChart width={700} height={280} data={Object.entries(stats.byKind).map(([k, v]) => ({
                    name: k.toUpperCase(),
                    count: v
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} stroke="#ff8c00" style={{ fontSize: '10px' }} />
                    <YAxis stroke="#ff8c00" style={{ fontSize: '10px' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#000', border: '1px solid #ff8c00', color: '#00d9ff' }}
                      labelStyle={{ color: '#ff8c00' }}
                    />
                    <Bar dataKey="count" fill="#ff8c00" />
                  </BarChart>
                </div>
              </div>

              <div className="border-2 border-orange-500 bg-black p-4">
                <div className="text-sm text-orange-400 mb-3 border-b border-orange-500 pb-2">
                  ORDER BOOK ANALYSIS
                </div>
                <div className="overflow-auto max-h-80">
                  <table className="w-full text-xs">
                    <thead className="text-orange-400 border-b border-orange-500">
                      <tr>
                        <th className="px-2 py-2 text-left">SYMBOL</th>
                        <th className="px-2 py-2 text-right">COUNT</th>
                        <th className="px-2 py-2 text-right">QTY</th>
                        <th className="px-2 py-2 text-right">AVG $</th>
                        <th className="px-2 py-2 text-right">ATR</th>
                        <th className="px-2 py-2 text-left">ACTION</th>
                      </tr>
                    </thead>
                    <tbody className="text-cyan-400">
                      {Object.entries(stats.ordersBySymbol).map(([symbol, data]) => (
                        <tr key={symbol} className="border-b border-orange-900 hover:bg-orange-950">
                          <td className="px-2 py-2 font-bold text-yellow-400">{symbol}</td>
                          <td className="px-2 py-2 text-right">{data.count}</td>
                          <td className="px-2 py-2 text-right">{data.total_qty}</td>
                          <td className="px-2 py-2 text-right">${data.avg_price.toFixed(2)}</td>
                          <td className="px-2 py-2 text-right">{data.avg_atr.toFixed(2)}</td>
                          <td className="px-2 py-2">
                            {Object.entries(data.actions).map(([action, count]) => (
                              <span key={action} className={`mr-2 px-1 ${action === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                                {action.toUpperCase()}:{count}
                              </span>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Timeline */}
            <div className="border-2 border-orange-500 bg-black p-4">
              <div className="text-sm text-orange-400 mb-3 border-b border-orange-500 pb-2 flex items-center justify-between">
                <span>NEWS PROCESSING TIMELINE [{stats.timelines.length} FLOWS]</span>
                <div className="flex gap-3 text-xs">
                  <span><span className="inline-block w-2 h-2 bg-blue-500 mr-1"></span>FILTER</span>
                  <span><span className="inline-block w-2 h-2 bg-green-500 mr-1"></span>DIRECTION</span>
                  <span><span className="inline-block w-2 h-2 bg-yellow-500 mr-1"></span>RISK</span>
                  <span><span className="inline-block w-2 h-2 bg-red-500 mr-1"></span>ORDER</span>
                </div>
              </div>
              
              <div className="space-y-2 max-h-96 overflow-auto">
                {stats.timelines.slice().reverse().slice(0, timelineLimit).map((timeline, idx) => (
                  <div key={idx} 
                       className="border border-orange-500 bg-black p-3 hover:border-cyan-400 cursor-pointer transition-colors"
                       onClick={() => setSelectedTimeline(selectedTimeline === idx ? null : idx)}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-1">
                          <span className="text-xs text-orange-400 font-mono">{formatTimestamp(timeline.ts)}</span>
                          {timeline.tickers.length > 0 && (
                            <span className="text-xs text-yellow-400 font-bold">
                              [{timeline.tickers.join(', ')}]
                            </span>
                          )}
                          {timeline.processing_time > 0 && (
                            <span className="text-xs text-cyan-400">
                              ⚡{timeline.processing_time.toFixed(0)}ms
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-cyan-400">{timeline.headline}</p>
                      </div>
                      <div className="flex gap-1 ml-3">
                        {timeline.events.map(e => e.kind).includes('llm_filter') && 
                          <span className="w-2 h-2 bg-blue-500" />}
                        {timeline.events.map(e => e.kind).includes('llm_direction') && 
                          <span className="w-2 h-2 bg-green-500" />}
                        {timeline.events.map(e => e.kind).includes('llm_risk') && 
                          <span className="w-2 h-2 bg-yellow-500" />}
                        {timeline.events.map(e => e.kind).includes('order_submitted') && 
                          <span className="w-2 h-2 bg-red-500" />}
                      </div>
                    </div>
                    
                    {selectedTimeline === idx && (
                      <div className="mt-3 pt-3 border-t border-orange-500 space-y-2">
                        {timeline.events.map((event, eidx) => (
                          <div key={eidx} className="text-xs bg-orange-950 bg-opacity-30 p-2 border-l-2 border-orange-500">
                            <div className="font-bold text-orange-400 mb-1">&gt; {event.kind.toUpperCase()}</div>
                            {event.kind === 'llm_filter' && event.payload?.result && (
                              <div className="text-cyan-400 space-y-0.5 ml-2">
                                <div>TYPE: {event.payload.result.type}</div>
                                <div>ACTIONABLE: {event.payload.result.actionable ? 'YES' : 'NO'}</div>
                                <div>REASON: {event.payload.result.reason}</div>
                                {event.payload.result.risk_flags?.length > 0 && (
                                  <div className="text-red-400">FLAGS: {event.payload.result.risk_flags.join(', ')}</div>
                                )}
                              </div>
                            )}
                            {event.kind === 'llm_direction' && event.payload?.result && (
                              <div className="text-cyan-400 space-y-0.5 ml-2">
                                <div>SENTIMENT: {event.payload.result.sentiment} | ACTION: {event.payload.result.action}</div>
                                <div>MAGNITUDE: {event.payload.result.magnitude} | CONFIDENCE: {event.payload.result.confidence}</div>
                                <div>VOLATILITY: {event.payload.result.volatility_factor}</div>
                                {event.payload.result.tickers_out?.length > 0 && (
                                  <div className="text-yellow-400">TARGETS: {event.payload.result.tickers_out.join(', ')}</div>
                                )}
                              </div>
                            )}
                            {event.kind === 'llm_risk' && event.payload?.result && (
                              <div className="text-cyan-400 space-y-0.5 ml-2">
                                <div className={event.payload.result.approve ? 'text-green-400' : 'text-red-400'}>
                                  APPROVED: {event.payload.result.approve ? 'YES' : 'NO'}
                                </div>
                                <div>ATR_MULT: {event.payload.result.atr_trail_mult} | TRAIL_EXTRA: {event.payload.result.trail_extra_pct}%</div>
                                <div>REASON: {event.payload.result.reason}</div>
                              </div>
                            )}
                            {event.kind === 'order_submitted' && event.payload && (
                              <div className="text-green-400 space-y-0.5 ml-2">
                                <div className="font-bold">
                                  {event.payload.symbol} | {event.payload.action.toUpperCase()} | QTY: {event.payload.qty} @ ${event.payload.last_price}
                                </div>
                                <div>ATR: {event.payload.atr} | WIDE_TRAIL: {event.payload.wide_trail} | TIGHT_TRAIL: {event.payload.tight_trail}</div>
                                <div>BREAKEVEN: {event.payload.breakeven_pct}% | POSITION: {event.payload.pos_pct}%</div>
                              </div>
                            )}
                            {event.kind === 'dropped_holdings' && event.payload && (
                              <div className="text-red-400 ml-2">
                                DROPPED: ALREADY HOLDING {event.payload.holdings?.join(', ')}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              {stats.timelines.length > timelineLimit && (
                <div className="mt-3 flex items-center justify-between text-xs text-orange-600">
                  <span>SHOWING {timelineLimit} OF {stats.timelines.length} FLOWS (newest first)</span>
                  <button
                    className="px-2 py-1 border border-orange-500 text-orange-400 hover:text-cyan-400 hover:border-cyan-400"
                    onClick={() => setTimelineLimit((n) => n + 20)}
                  >
                    LOAD MORE
                  </button>
                </div>
              )}
            </div>
          </>
        )}

        <div className="border-t-2 border-orange-500 mt-4 pt-3 text-center text-xs text-orange-600">
          METRICS ANALYZER v1.0 | LIVE POLLING | DATA FROM /metrics
        </div>
      </div>
    </div>
  );
};

export default MetricsAnalyzer;
