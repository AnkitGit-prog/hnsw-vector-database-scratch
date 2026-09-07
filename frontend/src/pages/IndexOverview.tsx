// IndexOverview.tsx — Index statistics dashboard
import { useEffect, useState } from 'react';
import { Database, Cpu, Layers, RefreshCw, Zap } from 'lucide-react';
import { getStats, updateEfSearch } from '../services/api';
import type { StatsResponse } from '../services/api';

function StatTile({ label, value, sub, color }: {
  label: string; value: string | number; sub?: string; color?: string;
}) {
  return (
    <div className="stat-tile">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function IndexOverview() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [newEf, setNewEf] = useState(50);
  const [efMsg, setEfMsg] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const s = await getStats();
      setStats(s);
      setNewEf(s.hnsw.ef_search);
    } catch {
      setError('Failed to load stats. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleEfUpdate = async () => {
    try {
      await updateEfSearch(newEf);
      setEfMsg(`ef_search updated to ${newEf}`);
      await load();
      setTimeout(() => setEfMsg(''), 3000);
    } catch {
      setEfMsg('Update failed.');
    }
  };

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '60px 0', color: 'var(--text-muted)' }}>
      <div className="spinner" /> Loading index statistics...
    </div>
  );

  if (error) return <div className="alert alert-error">{error}</div>;
  if (!stats) return null;

  const { exact, hnsw, embedder } = stats;

  return (
    <div>
      <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2>Index Overview</h2>
          <p>Real-time statistics for both vector indexes and the embedding model.</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}>
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* Status banner */}
      {stats.initialized ? (
        <div className="alert alert-success" style={{ marginBottom: 24 }}>
          ✓ Indexes loaded and ready. Both ExactVectorIndex and HNSWIndex are operational.
        </div>
      ) : (
        <div className="alert alert-info" style={{ marginBottom: 24 }}>
          ⚠ Indexes not initialized. Run <code>generate_dataset.py</code> then <code>build_index.py</code>.
        </div>
      )}

      {/* Exact Index */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon" style={{ background: 'var(--purple-dim)' }}>
              <Database size={16} color="var(--purple)" />
            </div>
            <span>Exact Brute-Force Index</span>
            <span className="badge badge-purple">Ground Truth</span>
          </div>
        </div>
        <div className="stat-grid">
          <StatTile label="Total Vectors" value={exact.total_vectors.toLocaleString()} />
          <StatTile label="Active Vectors" value={exact.active_vectors.toLocaleString()} color="var(--green)" />
          <StatTile label="Deleted Vectors" value={exact.deleted_vectors.toLocaleString()} color="var(--red)" sub="tombstoned" />
          <StatTile label="Dimension" value={exact.dimension ?? '—'} sub="float32" />
          <StatTile label="Build Time" value={exact.build_time_seconds ? `${exact.build_time_seconds.toFixed(2)}s` : '—'} />
          <StatTile label="Complexity" value="O(N·D)" sub="per query" />
        </div>
      </div>

      {/* HNSW Index */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon" style={{ background: 'var(--accent-dim)' }}>
              <Layers size={16} color="var(--accent)" />
            </div>
            <span>HNSW Approximate Index</span>
            <span className="badge badge-cyan">Graph Traversal</span>
          </div>
        </div>
        <div className="stat-grid">
          <StatTile label="Total Vectors" value={hnsw.total_vectors.toLocaleString()} />
          <StatTile label="Active Vectors" value={hnsw.active_vectors.toLocaleString()} color="var(--green)" />
          <StatTile label="Deleted Vectors" value={hnsw.deleted_vectors.toLocaleString()} color="var(--red)" sub="tombstoned" />
          <StatTile label="Dimension" value={hnsw.dimension ?? '—'} sub="float32" />
          <StatTile label="M (neighbors)" value={hnsw.M} sub="per layer" />
          <StatTile label="ef_construction" value={hnsw.ef_construction} sub="build quality" />
          <StatTile label="ef_search" value={hnsw.ef_search} sub="query accuracy" />
          <StatTile label="Max Layer" value={hnsw.max_layer ?? '—'} sub="graph layers" />
          <StatTile label="Total Edges" value={hnsw.total_edges?.toLocaleString() ?? '—'} />
          <StatTile label="Build Time" value={hnsw.build_time_seconds ? `${hnsw.build_time_seconds.toFixed(2)}s` : '—'} />
          <StatTile label="Entry Point" value={hnsw.entry_point?.slice(0, 10) ?? '—'} sub="global start node" />
          <StatTile label="Complexity" value="O(log N)" sub="empirical" />
        </div>

        {/* Live ef_search adjustment */}
        <div className="divider" />
        <div>
          <h4 style={{ marginBottom: 12, color: 'var(--text-secondary)' }}>
            <Zap size={14} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} />
            Live ef_search Adjustment
          </h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            Increase ef_search for higher recall (slower). Decrease for faster search (less accurate).
          </p>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">ef_search</label>
              <select className="input" style={{ width: 110 }} value={newEf} onChange={e => setNewEf(+e.target.value)}>
                {[10, 25, 50, 100, 200, 500].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
            <button className="btn btn-primary" onClick={handleEfUpdate}>
              Update ef_search
            </button>
            {efMsg && <span className="badge badge-green">{efMsg}</span>}
          </div>
        </div>
      </div>

      {/* Embedder */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon" style={{ background: 'var(--orange-dim)' }}>
              <Cpu size={16} color="var(--orange)" />
            </div>
            <span>Embedding Model</span>
          </div>
        </div>
        <div className="stat-grid">
          <StatTile label="Model" value={embedder.model} />
          <StatTile label="Dimension" value={embedder.dimension} />
          <StatTile label="Loaded" value={embedder.loaded ? 'Yes' : 'No'} color={embedder.loaded ? 'var(--green)' : 'var(--red)'} />
          <StatTile label="Load Time" value={embedder.load_time_seconds ? `${embedder.load_time_seconds.toFixed(2)}s` : '—'} />
        </div>
      </div>
    </div>
  );
}
