// App.tsx — Root component with sidebar navigation
import { useState } from 'react';
import {
  Search, Database, BarChart3, GitCompare, PlusCircle,
  Cpu, Layers, Activity
} from 'lucide-react';
import SemanticSearch from './pages/SemanticSearch';
import IndexOverview from './pages/IndexOverview';
import BenchmarkDashboard from './pages/BenchmarkDashboard';
import ComparisonView from './pages/ComparisonView';
import InsertDelete from './pages/InsertDelete';

type Page = 'search' | 'overview' | 'benchmark' | 'compare' | 'manage';

const NAV = [
  { id: 'search',    label: 'Semantic Search',    icon: Search,    section: 'Search' },
  { id: 'compare',   label: 'Compare Indexes',     icon: GitCompare, section: 'Search' },
  { id: 'overview',  label: 'Index Overview',      icon: Database,  section: 'Analytics' },
  { id: 'benchmark', label: 'Benchmark Dashboard', icon: BarChart3, section: 'Analytics' },
  { id: 'manage',    label: 'Insert / Delete',     icon: PlusCircle, section: 'Manage' },
] as const;

export default function App() {
  const [page, setPage] = useState<Page>('search');

  const renderPage = () => {
    switch (page) {
      case 'search':    return <SemanticSearch />;
      case 'overview':  return <IndexOverview />;
      case 'benchmark': return <BenchmarkDashboard />;
      case 'compare':   return <ComparisonView />;
      case 'manage':    return <InsertDelete />;
    }
  };

  const sections = [...new Set(NAV.map(n => n.section))];

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Layers size={18} color="#000" />
          </div>
          <h1>VectorDB</h1>
          <p>From Scratch · NumPy + HNSW</p>
        </div>

        {sections.map(section => (
          <div className="nav-section" key={section}>
            <div className="nav-section-title">{section}</div>
            {NAV.filter(n => n.section === section).map(item => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  className={`nav-item ${page === item.id ? 'active' : ''}`}
                  onClick={() => setPage(item.id as Page)}
                >
                  <Icon size={15} />
                  {item.label}
                </button>
              );
            })}
          </div>
        ))}

        <div style={{ marginTop: 'auto', padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Activity size={12} color="var(--green)" />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>API: localhost:8000</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}
