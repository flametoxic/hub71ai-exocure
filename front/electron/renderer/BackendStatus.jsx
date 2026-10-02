import React, { useEffect, useState } from 'react';
import { request } from './api-client.mjs';

export default function BackendStatus({ rid }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false, busy = false;
    const check = async () => {
      if (busy) return;
      busy = true;
      try {
        const [health, state] = await Promise.all([
          request('/health'),
          request('/state?rid=' + encodeURIComponent(rid)),
        ]);
        if (!disposed) { setResult({ health, state, receivedAt: new Date().toISOString() }); setError(''); }
      } catch (failure) {
        if (!disposed) { setResult(null); setError(failure.message); }
      } finally { busy = false; }
    };
    check();
    const timer = setInterval(check, 3000);
    return () => { disposed = true; clearInterval(timer); };
  }, [rid]);
  return <details className="card backend-status" data-status={error ? 'unavailable' : result ? 'connected' : 'loading'}>
    <summary>Backend: {error ? 'unavailable' : result ? 'connected' : 'checking…'}</summary>
    {result && <span className="language-mode"> · {result.health.language_mode === 'openai' ? 'AI' : 'local'}</span>}
    <div className="backend-details">{error ? <p role="alert">{error}</p> : result ? <>
      <p>Received: {result.receivedAt}</p>
      <div className="tag">GET /health and GET /twin/state</div>
      <pre className="json">{JSON.stringify(result, null, 2)}</pre>
    </> : <p>Waiting for backend response…</p>}</div>
  </details>;
}
