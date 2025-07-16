import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [userData, setUserData] = useState({});

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
  }, []);

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
            timestamp: new Date().toISOString(),
          }
        }),
      });

      // Send AI request
      const aiResponse = await fetch('/ai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
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

  return (
    <div className="App">
      <header className="App-header">
        <h1>AthenaMyst AI Demo</h1>
        <p>Experience the future of AI interaction</p>
      </header>
      
      <main className="App-main">
        <form onSubmit={handleSubmit} className="prompt-form">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask AthenaMyst anything..."
            className="prompt-input"
            rows={4}
          />
          <button 
            type="submit" 
            disabled={loading || !prompt.trim()}
            className="submit-button"
          >
            {loading ? 'Processing...' : 'Ask AthenaMyst'}
          </button>
        </form>

        {response && (
          <div className="response-container">
            <h3>AthenaMyst Response:</h3>
            <div className="response-text">{response}</div>
          </div>
        )}

        <div className="features">
          <h3>Features:</h3>
          <ul>
            <li>✨ Real-time AI interaction</li>
            <li>🔒 Secure data handling</li>
            <li>📊 Advanced analytics</li>
            <li>🚀 Lightning-fast responses</li>
          </ul>
        </div>
      </main>

      <footer className="App-footer">
        <p>© 2024 AthenaMyst Community - Experience the future of AI</p>
      </footer>
    </div>
  );
}

export default App; 