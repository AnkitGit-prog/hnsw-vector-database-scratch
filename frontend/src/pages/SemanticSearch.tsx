import { useState, useRef } from 'react';
import { Search, Clock } from 'lucide-react';
import { searchExact, searchHNSW } from '../services/api';
import type { SearchResult } from '../services/api';

const EXAMPLE_QUERIES = [
  "machine learning models need lots of training data",
  "renewable energy sources like solar and wind",
  "the human brain processes visual information",
  "ancient civilizations built large monuments",
  "cryptocurrency markets are highly volatile",
];

function SimilarityBar({ value }: { value: number }) {
  const val = typeof value === 'number' && !isNaN(value) ? value : 0;
  const pct = Math.max(0, Math.min(1, val)) * 100;
  return (
    <div className="sim-bar-wrap">
      <div className="sim-bar-bg">
        <div className="sim-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="sim-value">{val.toFixed(4)}</span>
    </div>
  );
}

function ResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  const meta = (result?.metadata || {}) as Record<string, unknown>;
  const text = typeof meta.text === 'string' ? meta.text : String(meta.text || '[no text]');
  const category = typeof meta.category === 'string' ? meta.category : null;
  const source = typeof meta.source === 'string' ? meta.source : null;
  return (
    <div className="result-card fade-in">
      <div className="result-rank">
        <span>#{rank}</span>
        {category && <span className="badge badge-purple">{category}</span>}
        {source && <span className="badge badge-cyan">{source}</span>}
        <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono', fontSize: '0.65rem', color: 'var(--text-muted)' }}>{result?.id || `res_${rank}`}</span>
      </div>
      <div className="result-text">{text}</div>
      <SimilarityBar value={result?.similarity} />
    </div>
  );
}

export default function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'hnsw' | 'exact'>('hnsw');
  const [topK, setTopK] = useState(10);
  const [efSearch, setEfSearch] = useState(50);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSearch = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? query).trim();
    if (!q) return;
    setLoading(true);
    setError('');
    try {
      const res = mode === 'hnsw'
        ? await searchHNSW(q, topK, efSearch)
        : await searchExact(q, topK);
      setResults(res.results);
      setLatency(res.latency_ms);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Search failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (q: string) => {
    setQuery(q);
    inputRef.current?.focus();
    handleSearch(q);
  };

  return (
    <div>
      <div className="section-header">
        <h2>Semantic Search</h2>
        <p>Enter any natural language statement to find the best matching stored texts using vector similarity.</p>
      </div>

      {/* Search card */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          {/* Mode toggle */}
          <div className="toggle-group" style={{ width: 240 }}>
            <button className={`toggle-btn ${mode === 'hnsw' ? 'active' : ''}`} onClick={() => setMode('hnsw')}>
              ⚡ HNSW
            </button>
            <button className={`toggle-btn ${mode === 'exact' ? 'active' : ''}`} onClick={() => setMode('exact')}>
              🎯 Exact
            </button>
          </div>

          {/* top-k */}
          <div className="form-group">
            <label className="form-label">Top-K</label>
            <select className="input" style={{ width: 80 }} value={topK} onChange={e => setTopK(+e.target.value)}>
              {[5, 10, 20, 50].map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>

          {/* ef_search — only for HNSW */}
          {mode === 'hnsw' && (
            <div className="form-group">
              <label className="form-label">ef_search</label>
              <select className="input" style={{ width: 100 }} value={efSearch} onChange={e => setEfSearch(+e.target.value)}>
                {[10, 25, 50, 100, 200].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          )}
        </div>

        {/* Query input */}
        <textarea
          ref={inputRef}
          className="input"
          style={{ marginBottom: 12, minHeight: 72 }}
          placeholder="Enter a natural language query, e.g. 'neural networks learn from examples'..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch(); } }}
        />

        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button className="btn btn-primary" onClick={handleSearch} disabled={loading || !query.trim()}>
            {loading ? <div className="spinner" /> : <Search size={15} />}
            {loading ? 'Searching...' : 'Search'}
          </button>

          {latency !== null && (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Clock size={12} />
              {latency.toFixed(2)}ms
              <span className={`badge ${mode === 'hnsw' ? 'badge-cyan' : 'badge-purple'}`} style={{ marginLeft: 4 }}>
                {mode === 'hnsw' ? 'HNSW' : 'Exact'}
              </span>
            </span>
          )}
        </div>

        {/* Example queries */}
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Example queries
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {EXAMPLE_QUERIES.map((q, i) => (
              <button key={i} className="btn btn-ghost btn-sm" onClick={() => handleExample(q)}>
                {q.length > 40 ? q.slice(0, 40) + '…' : q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

      {/* Results */}
      {results.length > 0 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <h3 style={{ color: 'var(--text-primary)' }}>Results</h3>
            <span className="badge badge-green">{results.length} matches</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Query: &ldquo;<em>{query.slice(0, 80)}{query.length > 80 ? '…' : ''}</em>&rdquo;
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {results.map((r, i) => (
              <ResultCard key={r.id} result={r} rank={i + 1} />
            ))}
          </div>
        </div>
      )}

      {results.length === 0 && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
          <Search size={48} strokeWidth={1} style={{ marginBottom: 16, opacity: 0.3 }} />
          <p>Enter a query above to find semantically similar texts</p>
        </div>
      )}
    </div>
  );
}
