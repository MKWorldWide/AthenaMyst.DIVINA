import React, { useState } from 'react';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const sendPrompt = async () => {
    setLoading(true);
    setResponse('');
    try {
      const res = await fetch('http://localhost:4000/ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setResponse(data.result);
    } catch (err) {
      setResponse('Error contacting AthenaMyst-Test API.');
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 500, margin: '40px auto', fontFamily: 'Inter, sans-serif' }}>
      <h1>AthenaMyst-Test Public Demo</h1>
      <p>Interact with the public Athena AI interface. No real trading or privileged logic is exposed.</p>
      <input
        type="text"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        placeholder="Type your prompt..."
        style={{ width: '100%', padding: 8, fontSize: 16 }}
      />
      <button onClick={sendPrompt} style={{ marginTop: 12, padding: '8px 16px', fontSize: 16 }} disabled={loading}>
        {loading ? 'Sending...' : 'Send to Athena'}
      </button>
      {response && (
        <div style={{ marginTop: 24, background: '#f4f4f4', padding: 16, borderRadius: 8 }}>
          <strong>Response:</strong>
          <div>{response}</div>
        </div>
      )}
    </div>
  );
} 