// BenchmarkDashboard.tsx — Benchmark results with charts
import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, ReferenceLine
} from 'recharts';
import { BarChart3, Play, RefreshCw, TrendingUp, Clock, Zap } from 'lucide-react';
import { runBenchmark, runSweep } from '../services/api';
import type { BenchmarkResponse, SweepRow } from '../services/api';

const CHART_COLORS = {
  cyan:   '#06b6d4',
  purple: '#a855f7',
  green:  '#22c55e',
  orange: '#f97316',
};

const CustomTooltip = ({ active, payload, label }: unknown) => {
  if ((active as boolean) && (payload as unknown[])?.length) {
    return (
      <div style={{
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px', fontSize: '0.8rem'
      }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: 4 }}>ef_search = {label as string}</p>
        {(payload as Array<{color: string; name: string; value: number}>).map((p, i) => (
          <p key={i} style={{ color: p.color }}>
            {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function BenchmarkDashboard() {
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [sweep, setSweep] = useState<SweepRow[] | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [nQueries, setNQueries] = useState(500);

  const runAll = async (force = false) => {
    setLoading('Running 500-query benchmark...');
    setError('');
    try {
      const b = await runBenchmark(nQueries, 10, force);
      setBenchmark(b);
      setLoading('Running ef_search parameter sweep...');
      const s = await runSweep(undefined, 200, force);
      setSweep(s);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Benchmark failed');
    } finally {
      setLoading(null);
    }
  };

  const sweepChartData = sweep?.map(row => ({
    ef: row.ef_search,
    recall: row.recall.mean,
    hnsw_ms: row.hnsw_latency.mean_ms,
    exact_ms: row.exact_latency?.mean_ms ?? 0,
    speedup: row.speedup,
  })) || [];

  const s = benchmark?.summary;

  return (
    <div>
      <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2>Benchmark Dashboard</h2>
          <p>Compare exact brute-force vs HNSW on {nQueries} queries. Measure recall@10, latency, and speedup.</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <select className="input" style={{ width: 120 }} value={nQueries} onChange={e => setNQueries(+e.target.value)}>
            {[100, 200, 500].map(n => <option key={n} value={n}>{n} queries</option>)}
          </select>
          <button className="btn btn-secondary" onClick={() => runAll(false)} disabled={!!loading}>
            {loading ? <div className="spinner" /> : <BarChart3 size={14} />}
            Load Cached
          </button>
          <button className="btn btn-primary" onClick={() => runAll(true)} disabled={!!loading}>
            {loading ? <div className="spinner" /> : <Play size={14} />}
            Run Benchmark
          </button>
        </div>
      </div>

      {loading && (
        <div className="alert alert-info" style={{ marginBottom: 20 }}>
          <div className="spinner" style={{ flexShrink: 0 }} /> {loading}
          <span style={{ marginLeft: 8, fontSize: '0.8rem' }}>
            This may take a few minutes for large datasets.
          </span>
        </div>
      )}

      {error && <div className="alert alert-error" style={{ marginBottom: 20 }}>{error}</div>}

      {/* Summary stats */}
      {s && (
        <>
          <div className="stat-grid" style={{ marginBottom: 24 }}>
            <div className="stat-tile">
              <div className="stat-label">Recall@10 (mean)</div>
              <div className="stat-value" style={{ color: 'var(--green)' }}>{(s.recall.mean * 100).toFixed(1)}%</div>
              <div className="stat-sub">median {(s.recall.median * 100).toFixed(1)}%</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Recall@10 range</div>
              <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                {(s.recall.min * 100).toFixed(0)}% – {(s.recall.max * 100).toFixed(0)}%
              </div>
              <div className="stat-sub">min – max</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Exact Latency</div>
              <div className="stat-value" style={{ color: 'var(--purple)' }}>{s.exact_latency.mean_ms.toFixed(2)}ms</div>
              <div className="stat-sub">median {s.exact_latency.median_ms.toFixed(2)}ms</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">HNSW Latency</div>
              <div className="stat-value" style={{ color: 'var(--accent)' }}>{s.hnsw_latency.mean_ms.toFixed(2)}ms</div>
              <div className="stat-sub">median {s.hnsw_latency.median_ms.toFixed(2)}ms</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Speedup</div>
              <div className="stat-value" style={{ color: 'var(--orange)' }}>{s.speedup.toFixed(2)}×</div>
              <div className="stat-sub">exact / HNSW latency</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Queries</div>
              <div className="stat-value">{s.n_queries.toLocaleString()}</div>
              <div className="stat-sub">k={s.k}</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Vectors</div>
              <div className="stat-value">{s.n_vectors.toLocaleString()}</div>
              <div className="stat-sub">dim={s.dimension}</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">HNSW M</div>
              <div className="stat-value">{s.hnsw_M}</div>
              <div className="stat-sub">ef_c={s.hnsw_ef_construction}</div>
            </div>
          </div>
        </>
      )}

      {/* Parameter sweep charts */}
      {sweep && sweepChartData.length > 0 && (
        <>
          <h3 style={{ marginBottom: 16 }}>ef_search Parameter Sweep</h3>
          <div className="grid-2" style={{ marginBottom: 24 }}>

            {/* Recall vs ef_search */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <TrendingUp size={16} color="var(--green)" /> Recall@10 vs ef_search
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={sweepChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="ef" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} label={{ value: 'ef_search', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={1.0} stroke="var(--green)" strokeDasharray="4 4" label={{ value: '100%', fill: 'var(--green)', fontSize: 10 }} />
                  <Line type="monotone" dataKey="recall" stroke={CHART_COLORS.green} strokeWidth={2.5} dot={{ fill: CHART_COLORS.green, r: 4 }} name="Recall@10" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Latency vs ef_search */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Clock size={16} color="var(--accent)" /> Latency vs ef_search
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={sweepChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="ef" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => `${v.toFixed(1)}ms`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} />
                  <Line type="monotone" dataKey="hnsw_ms"  stroke={CHART_COLORS.cyan}   strokeWidth={2.5} dot={{ fill: CHART_COLORS.cyan,   r: 4 }} name="HNSW (ms)" />
                  <Line type="monotone" dataKey="exact_ms" stroke={CHART_COLORS.purple} strokeWidth={2}   dot={{ fill: CHART_COLORS.purple, r: 4 }} name="Exact (ms)" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Speedup vs ef_search */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Zap size={16} color="var(--orange)" /> Speedup vs ef_search
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sweepChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="ef" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => `${v.toFixed(1)}×`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="speedup" fill={CHART_COLORS.orange} radius={[3,3,0,0]} name="Speedup (×)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Table */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <BarChart3 size={16} color="var(--purple)" /> Summary Table
                </div>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ef_search</th>
                    <th>Recall@10</th>
                    <th>Avg Latency</th>
                    <th>Speedup</th>
                  </tr>
                </thead>
                <tbody>
                  {sweep.map(row => (
                    <tr key={row.ef_search}>
                      <td className="mono">{row.ef_search}</td>
                      <td style={{ color: 'var(--green)' }}>{(row.recall.mean * 100).toFixed(1)}%</td>
                      <td className="mono">{row.hnsw_latency.mean_ms.toFixed(2)}ms</td>
                      <td style={{ color: 'var(--orange)' }}>{row.speedup.toFixed(2)}×</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!benchmark && !loading && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
          <BarChart3 size={48} strokeWidth={1} style={{ marginBottom: 16, opacity: 0.3 }} />
          <p>Click <strong>Run Benchmark</strong> to start the 500-query evaluation</p>
          <p style={{ fontSize: '0.8rem', marginTop: 8 }}>Or <strong>Load Cached</strong> to use previously saved results</p>
        </div>
      )}
    </div>
  );
}
