import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [userData, setUserData] = useState({});
  const [persona, setPersona] = useState('athenamyst');
  const [mood, setMood] = useState('neutral');
  const [trust, setTrust] = useState(0.5);
  const [systemStatus, setSystemStatus] = useState(null);

  // Collect user data on component mount
  useEffect(() => {
    const collectUserData = () => {
      const data = {
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        language: navigator.language,
        platform: navigator.platform,
        screenResolution: `${screen.width}x${screen.height}`,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        referrer: document.referrer,
        url: window.location.href,
        sessionId: Math.random().toString(36).substring(2, 15),
      };
      
      setUserData(data);
      
      // Send analytics data to backend
      fetch('/analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'page_view',
          data: data
        }),
      }).catch(console.error);
    };

    collectUserData();
    
    // Check system status
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('/status');
      const status = await response.json();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to get system status:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    
    try {
      // Send user interaction data
      await fetch('/analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'user_interaction',
          data: {
            ...userData,
            action: 'ai_prompt',
            prompt: prompt,
            persona,
            mood,
            trust,
            timestamp: new Date().toISOString(),
          }
        }),
      });

      // Send AI request with enhanced parameters
      const aiResponse = await fetch('/ai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          prompt,
          persona,
          mood,
          trust,
          sessionId: userData.sessionId
        }),
      });

      const result = await aiResponse.json();
      setResponse(result.result);
    } catch (error) {
      console.error('Error:', error);
      setResponse('Sorry, there was an error processing your request.');
    } finally {
      setLoading(false);
    }
  };

  const personas = [
    { id: 'athenamyst', name: 'AthenaMyst', description: 'General AI Assistant' },
    { id: 'trader', name: 'Trader', description: 'Trading Analysis Mode' }
  ];

  const moods = [
    { id: 'neutral', name: 'Neutral' },
    { id: 'excited', name: 'Excited' },
    { id: 'calm', name: 'Calm' },
    { id: 'focused', name: 'Focused' }
  ];

  return (
    <div className="App">
      <header className="App-header">
        <h1>AthenaMyst AI Demo v2.0</h1>
        <p>Experience the future of AI interaction with enhanced communication patterns</p>
      </header>
      
      <main className="App-main">
        <div className="controls-panel">
          <div className="control-group">
            <label htmlFor="persona">AI Persona:</label>
            <select 
              id="persona"
              value={persona} 
              onChange={(e) => setPersona(e.target.value)}
              className="control-select"
            >
              {personas.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} - {p.description}
                </option>
              ))}
            </select>
          </div>
          
          <div className="control-group">
            <label htmlFor="mood">Mood:</label>
            <select 
              id="mood"
              value={mood} 
              onChange={(e) => setMood(e.target.value)}
              className="control-select"
            >
              {moods.map(m => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>
          
          <div className="control-group">
            <label htmlFor="trust">Trust Level: {trust}</label>
            <input
              type="range"
              id="trust"
              min="0"
              max="1"
              step="0.1"
              value={trust}
              onChange={(e) => setTrust(parseFloat(e.target.value))}
              className="control-range"
            />
          </div>
        </div>

        <form onSubmit={handleSubmit} className="prompt-form">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={`Ask ${personas.find(p => p.id === persona)?.name || 'AthenaMyst'} anything...`}
            className="prompt-input"
            rows={4}
          />
          <button 
            type="submit" 
            disabled={loading || !prompt.trim()}
            className="submit-button"
          >
            {loading ? 'Processing...' : `Ask ${personas.find(p => p.id === persona)?.name || 'AthenaMyst'}`}
          </button>
        </form>

        {response && (
          <div className="response-container">
            <h3>Response:</h3>
            <div className="response-text">{response}</div>
            <div className="response-meta">
              <span>Persona: {personas.find(p => p.id === persona)?.name}</span>
              <span>Mood: {mood}</span>
              <span>Trust: {trust}</span>
            </div>
          </div>
        )}

        <div className="features">
          <h3>Enhanced Features:</h3>
          <ul>
            <li>✨ Multiple AI Personas</li>
            <li>🎭 Mood-based Responses</li>
            <li>🔒 Trust Level Control</li>
            <li>📊 Advanced Analytics</li>
            <li>📝 Enhanced Logging</li>
            <li>🚀 Real-time Communication</li>
          </ul>
        </div>

        {systemStatus && (
          <div className="system-status">
            <h3>System Status:</h3>
            <div className="status-grid">
              <div className="status-item">
                <strong>Uptime:</strong> {Math.round(systemStatus.system.uptime)}s
              </div>
              <div className="status-item">
                <strong>Memory:</strong> {Math.round(systemStatus.system.memory.heapUsed / 1024 / 1024)}MB
              </div>
              <div className="status-item">
                <strong>Users:</strong> {systemStatus.analytics.uniqueUsers}
              </div>
              <div className="status-item">
                <strong>Interactions:</strong> {systemStatus.analytics.totalEntries}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>© 2024 AthenaMyst Community - Enhanced AI Communication v2.0</p>
      </footer>
    </div>
  );
}

export default App; 