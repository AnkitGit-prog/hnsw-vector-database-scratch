// api.ts — Typed API client for all backend endpoints

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: BASE_URL, timeout: 120000 });

api.interceptors.response.use(
  res => res,
  err => {
    const detail = err.response?.data?.detail;
    if (detail) {
      return Promise.reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)));
    }
    return Promise.reject(err);
  }
);

// ---- Types ----

export interface SearchResult {
  id: string;
  similarity: number;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  results: SearchResult[];
  latency_ms: number;
  index_type: string;
  n_results: number;
  top_k: number;
  ef_search?: number;
}

export interface StatsResponse {
  initialized: boolean;
  exact: {
    type: string;
    total_vectors: number;
    active_vectors: number;
    deleted_vectors: number;
    dimension: number;
    build_time_seconds: number;
    insert_count: number;
  };
  hnsw: {
    type: string;
    total_vectors: number;
    active_vectors: number;
    deleted_vectors: number;
    dimension: number;
    M: number;
    ef_construction: number;
    ef_search: number;
    max_layer: number;
    entry_point: string;
    total_edges: number;
    build_time_seconds: number;
    insert_count: number;
  };
  embedder: {
    model: string;
    dimension: number;
    load_time_seconds: number;
    loaded: boolean;
  };
}

export interface BenchmarkSummary {
  n_queries: number;
  k: number;
  n_vectors: number;
  dimension: number;
  ef_search_used: number;
  hnsw_M: number;
  hnsw_ef_construction: number;
  recall: { mean: number; median: number; min: number; max: number; std: number };
  exact_latency: { mean_ms: number; median_ms: number; min_ms: number; max_ms: number; p95_ms: number };
  hnsw_latency:  { mean_ms: number; median_ms: number; min_ms: number; max_ms: number; p95_ms: number };
  speedup: number;
  build_time_exact_s?: number;
  build_time_hnsw_s?: number;
}

export interface BenchmarkResponse {
  summary: BenchmarkSummary;
  per_query: Array<{
    query_index: number;
    exact_ids: string[];
    hnsw_ids: string[];
    recall_at_k: number;
    exact_latency_ms: number;
    hnsw_latency_ms: number;
    exact_top_similarities: number[];
    hnsw_top_similarities: number[];
  }>;
}

export interface SweepRow {
  ef_search: number;
  recall: { mean: number; median: number };
  exact_latency: { mean_ms: number };
  hnsw_latency: { mean_ms: number };
  speedup: number;
}

// ---- API Functions ----

export const healthCheck = () => api.get('/health').then(r => r.data);

export const getStats = (): Promise<StatsResponse> =>
  api.get('/stats').then(r => r.data);

export const searchExact = (
  query: string | number[],
  topK: number = 10
): Promise<SearchResponse> =>
  api.post('/search/exact', {
    ...(typeof query === 'string' ? { text: query } : { vector: query }),
    top_k: topK,
  }).then(r => r.data);

export const searchHNSW = (
  query: string | number[],
  topK: number = 10,
  efSearch?: number
): Promise<SearchResponse> =>
  api.post('/search/hnsw', {
    ...(typeof query === 'string' ? { text: query } : { vector: query }),
    top_k: topK,
    ef_search: efSearch,
  }).then(r => r.data);

export const insertVector = (
  id: string,
  text: string,
  metadata?: Record<string, unknown>
) =>
  api.post('/vectors', { id, text, metadata: metadata || {} }).then(r => r.data);

export const deleteVector = (id: string) =>
  api.delete(`/vectors/${id}`).then(r => r.data);

export const getVector = (id: string) =>
  api.get(`/vectors/${id}`).then(r => r.data);

export const runBenchmark = (
  nQueries: number = 500,
  k: number = 10,
  force: boolean = false
): Promise<BenchmarkResponse> =>
  api.post('/benchmark', { n_queries: nQueries, k, force }).then(r => r.data);

export const runSweep = (
  efValues?: number[],
  nQueries: number = 200,
  force: boolean = false
): Promise<SweepRow[]> =>
  api.post('/benchmark/sweep', {
    ef_search_values: efValues,
    n_queries: nQueries,
    force,
  }).then(r => r.data);

export const updateEfSearch = (efSearch: number) =>
  api.put('/config/ef_search', { ef_search: efSearch }).then(r => r.data);
