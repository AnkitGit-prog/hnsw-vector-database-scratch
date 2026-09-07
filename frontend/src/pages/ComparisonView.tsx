// ComparisonView.tsx — Side-by-side exact vs HNSW result comparison
import { useState } from 'react';
import { GitCompare, Search } from 'lucide-react';
import { searchExact, searchHNSW } from '../services/api';
import type { SearchResult } from '../services/api';

function ResultCol({
  title,
  color,
  results,
  otherIds,
  badge,
}: {
  title: string;
  color: string;
  badge: string;
  results: SearchResult[];
  otherIds: Set<string>;
}) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <h4 style={{ color }}>{title}</h4>
        <span className={`badge ${badge}`}>{results.length} results</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {results.map((r, i) => {
          const meta = r.metadata as Record<string, string>;
          const inOther = otherIds.has(r.id);
          let matchClass = '';
          let matchLabel = '';
          if (title.includes('Exact')) {
            matchClass = inOther ? 'match-exact' : 'match-missed';
            matchLabel = inOther ? '✓ in HNSW' : '✗ missed by HNSW';
          } else {
            matchClass = inOther ? 'match-exact' : 'match-approx-only';
            matchLabel = inOther ? '✓ in Exact' : '★ HNSW-only';
          }
          return (
            <div key={r.id} className={`result-card ${matchClass}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>#{i + 1} · {r.id}</span>
                <span style={{
                  fontSize: '0.65rem',
                  fontWeight: 700,
                  color: matchClass === 'match-exact' ? 'var(--green)'
                    : matchClass === 'match-missed' ? 'var(--red)'
                    : 'var(--yellow)',
                }}>
                  {matchLabel}
                </span>
              </div>
              <div className="result-text" style={{ fontSize: '0.82rem', WebkitLineClamp: 3, display: '-webkit-box', WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                {meta.text || '[no text]'}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
                <div className="sim-bar-bg" style={{ flex: 1 }}>
                  <div className="sim-bar-fill" style={{ width: `${r.similarity * 100}%`, background: color }} />
                </div>
                <span className="sim-value">{r.similarity.toFixed(4)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ComparisonView() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [efSearch, setEfSearch] = useState(50);
  const [exactResults, setExactResults] = useState<SearchResult[]>([]);
  const [hnswResults, setHnswResults] = useState<SearchResult[]>([]);
  const [latencyExact, setLatencyExact] = useState<number | null>(null);
  const [latencyHnsw, setLatencyHnsw] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searched, setSearched] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    try {
      const [eRes, hRes] = await Promise.all([
        searchExact(query, topK),
        searchHNSW(query, topK, efSearch),
      ]);
      setExactResults(eRes.results);
      setHnswResults(hRes.results);
      setLatencyExact(eRes.latency_ms);
      setLatencyHnsw(hRes.latency_ms);
      setSearched(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const exactIds = new Set(exactResults.map(r => r.id));
  const hnswIds  = new Set(hnswResults.map(r => r.id));
  const overlap  = [...exactIds].filter(id => hnswIds.has(id)).length;
  const recall   = topK > 0 ? overlap / topK : 0;

  return (
    <div>
      <div className="section-header">
        <h2>Index Comparison</h2>
        <p>Run the same query against both indexes and see exactly what HNSW gets right and what it misses.</p>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="form-row" style={{ flexWrap: 'wrap', marginBottom: 12 }}>
          <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
            <label className="form-label">Query</label>
            <input
              className="input"
              placeholder="Enter your query..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Top-K</label>
            <select className="input" style={{ width: 80 }} value={topK} onChange={e => setTopK(+e.target.value)}>
              {[5, 10, 20].map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">ef_search</label>
            <select className="input" style={{ width: 100 }} value={efSearch} onChange={e => setEfSearch(+e.target.value)}>
              {[10, 25, 50, 100, 200].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <button className="btn btn-primary" onClick={handleSearch} disabled={loading || !query.trim()} style={{ alignSelf: 'flex-end' }}>
            {loading ? <div className="spinner" /> : <Search size={14} />}
            Compare
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

      {/* Recall summary */}
      {searched && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'center' }}>
            <div>
              <div className="stat-label">Recall@{topK}</div>
              <div className="stat-value" style={{ color: recall >= 0.9 ? 'var(--green)' : recall >= 0.7 ? 'var(--yellow)' : 'var(--red)' }}>
                {(recall * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div className="stat-label">Common Results</div>
              <div className="stat-value">{overlap} / {topK}</div>
            </div>
            <div>
              <div className="stat-label">Exact Latency</div>
              <div className="stat-value" style={{ color: 'var(--purple)', fontSize: '1.2rem' }}>{latencyExact?.toFixed(2)}ms</div>
            </div>
            <div>
              <div className="stat-label">HNSW Latency</div>
              <div className="stat-value" style={{ color: 'var(--accent)', fontSize: '1.2rem' }}>{latencyHnsw?.toFixed(2)}ms</div>
            </div>
            {latencyExact && latencyHnsw && (
              <div>
                <div className="stat-label">Speedup</div>
                <div className="stat-value" style={{ color: 'var(--orange)' }}>{(latencyExact / latencyHnsw).toFixed(2)}×</div>
              </div>
            )}
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              <span style={{ color: 'var(--green)', fontWeight: 600 }}>Green border</span> = in both &nbsp;|&nbsp;
              <span style={{ color: 'var(--red)', fontWeight: 600 }}>Red border</span> = exact-only &nbsp;|&nbsp;
              <span style={{ color: 'var(--yellow)', fontWeight: 600 }}>Yellow border</span> = HNSW-only
            </div>
          </div>
        </div>
      )}

      {/* Side-by-side */}
      {searched && (
        <div className="grid-2">
          <ResultCol
            title="🎯 Exact Brute-Force (Ground Truth)"
            color="var(--purple)"
            badge="badge-purple"
            results={exactResults}
            otherIds={hnswIds}
          />
          <ResultCol
            title="⚡ HNSW Approximate"
            color="var(--accent)"
            badge="badge-cyan"
            results={hnswResults}
            otherIds={exactIds}
          />
        </div>
      )}

      {!searched && !loading && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
          <GitCompare size={48} strokeWidth={1} style={{ marginBottom: 16, opacity: 0.3 }} />
          <p>Enter a query to see a side-by-side comparison of both indexes</p>
        </div>
      )}
    </div>
  );
}
