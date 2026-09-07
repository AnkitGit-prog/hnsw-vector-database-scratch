// InsertDelete.tsx — Insert and delete vectors
import { useState } from 'react';
import { PlusCircle, Trash2, Search } from 'lucide-react';
import { insertVector, deleteVector, getVector } from '../services/api';

let insertCounter = 1000;

export default function InsertDelete() {
  // Insert state
  const [insertText, setInsertText] = useState('');
  const [customId, setCustomId] = useState('');
  const [insertMsg, setInsertMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [inserting, setInserting] = useState(false);

  // Delete state
  const [deleteId, setDeleteId] = useState('');
  const [deleteMsg, setDeleteMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Lookup state
  const [lookupId, setLookupId] = useState('');
  const [lookupResult, setLookupResult] = useState<Record<string, unknown> | null>(null);
  const [lookupMsg, setLookupMsg] = useState('');

  const handleInsert = async () => {
    if (!insertText.trim()) return;
    setInserting(true);
    setInsertMsg(null);
    const id = customId.trim() || `user_${++insertCounter}_${Date.now()}`;
    try {
      const res = await insertVector(id, insertText);
      setInsertMsg({ type: 'success', text: `✓ Inserted "${id}" (dim=${res.dimension})` });
      setInsertText('');
      setCustomId('');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Insert failed';
      setInsertMsg({ type: 'error', text: msg });
    } finally {
      setInserting(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteId.trim()) return;
    setDeleting(true);
    setDeleteMsg(null);
    try {
      const res = await deleteVector(deleteId.trim());
      setDeleteMsg({ type: 'success', text: `✓ Deleted "${res.id}" — tombstone applied to both indexes.` });
      setDeleteId('');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Delete failed';
      setDeleteMsg({ type: 'error', text: msg });
    } finally {
      setDeleting(false);
    }
  };

  const handleLookup = async () => {
    if (!lookupId.trim()) return;
    setLookupResult(null);
    setLookupMsg('');
    try {
      const res = await getVector(lookupId.trim());
      setLookupResult(res);
    } catch {
      setLookupMsg(`Vector "${lookupId}" not found.`);
    }
  };

  return (
    <div>
      <div className="section-header">
        <h2>Insert / Delete Vectors</h2>
        <p>Add new texts or remove existing vectors from both indexes simultaneously.</p>
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>

        {/* Insert */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <div className="card-icon" style={{ background: 'var(--green-dim)' }}>
                <PlusCircle size={16} color="var(--green)" />
              </div>
              Insert New Vector
            </div>
          </div>
          <p style={{ fontSize: '0.83rem', color: 'var(--text-muted)', marginBottom: 16 }}>
            Enter any text. It will be embedded using <code style={{ color: 'var(--accent)' }}>all-MiniLM-L6-v2</code>
            and inserted into both ExactIndex and HNSWIndex.
          </p>

          <div className="form-group" style={{ marginBottom: 12 }}>
            <label className="form-label">Text</label>
            <textarea
              className="input"
              placeholder="Enter the text to embed and insert..."
              value={insertText}
              onChange={e => setInsertText(e.target.value)}
              style={{ minHeight: 90 }}
            />
          </div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label className="form-label">Custom ID (optional)</label>
            <input
              className="input"
              placeholder="Auto-generated if empty"
              value={customId}
              onChange={e => setCustomId(e.target.value)}
            />
          </div>

          <button className="btn btn-primary" onClick={handleInsert} disabled={inserting || !insertText.trim()}>
            {inserting ? <div className="spinner" /> : <PlusCircle size={14} />}
            {inserting ? 'Embedding & Inserting...' : 'Insert'}
          </button>

          {insertMsg && (
            <div className={`alert ${insertMsg.type === 'success' ? 'alert-success' : 'alert-error'}`} style={{ marginTop: 12 }}>
              {insertMsg.text}
            </div>
          )}
        </div>

        {/* Delete */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <div className="card-icon" style={{ background: 'var(--red-dim)' }}>
                <Trash2 size={16} color="var(--red)" />
              </div>
              Delete Vector (Tombstone)
            </div>
          </div>
          <div className="alert alert-info" style={{ marginBottom: 16, fontSize: '0.8rem' }}>
            <span style={{ fontWeight: 600 }}>Tombstone deletion:</span> The vector is logically deleted from both indexes.
            In HNSW, graph edges are preserved to maintain connectivity.
            The node is excluded from all future search results.
          </div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label className="form-label">Vector ID</label>
            <input
              className="input"
              placeholder="e.g. vec_000042 or your custom ID"
              value={deleteId}
              onChange={e => setDeleteId(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleDelete()}
            />
          </div>

          <button className="btn btn-danger" onClick={handleDelete} disabled={deleting || !deleteId.trim()}>
            {deleting ? <div className="spinner" /> : <Trash2 size={14} />}
            {deleting ? 'Deleting...' : 'Delete'}
          </button>

          {deleteMsg && (
            <div className={`alert ${deleteMsg.type === 'success' ? 'alert-success' : 'alert-error'}`} style={{ marginTop: 12 }}>
              {deleteMsg.text}
            </div>
          )}
        </div>
      </div>

      {/* Lookup */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon" style={{ background: 'var(--accent-dim)' }}>
              <Search size={16} color="var(--accent)" />
            </div>
            Lookup Vector by ID
          </div>
        </div>
        <div className="form-row" style={{ marginBottom: 12 }}>
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="e.g. vec_000001"
            value={lookupId}
            onChange={e => setLookupId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleLookup()}
          />
          <button className="btn btn-secondary" onClick={handleLookup} disabled={!lookupId.trim()}>
            <Search size={14} /> Lookup
          </button>
        </div>

        {lookupMsg && <div className="alert alert-error">{lookupMsg}</div>}
        {lookupResult && (
          <div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <span className="badge badge-cyan">{(lookupResult as Record<string, unknown>).id as string}</span>
              {(lookupResult as Record<string, unknown>).deleted && <span className="badge badge-red">DELETED</span>}
            </div>
            <div className="code-block" style={{ color: 'var(--text-secondary)', maxHeight: 300, overflowY: 'auto' }}>
              {JSON.stringify(lookupResult, null, 2)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
