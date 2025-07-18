const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public/dist')));

// Enhanced logging system
const logDir = path.join(__dirname, 'logs');
const analyticsLogPath = path.join(logDir, 'analytics.log');
const interactionLogPath = path.join(logDir, 'interactions.log');

// Ensure log directory exists
async function ensureLogDirectory() {
  try {
    await fs.mkdir(logDir, { recursive: true });
  } catch (error) {
    console.error('Failed to create log directory:', error.message);
  }
}

// Enhanced logging function
async function logInteraction(type, data) {
  try {
    await ensureLogDirectory();
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      type,
      data,
      sessionId: data.sessionId || 'unknown'
    };
    
    const logLine = `[${timestamp}] ${type.toUpperCase()}: ${JSON.stringify(logEntry)}\n`;
    await fs.appendFile(interactionLogPath, logLine);
    
    console.log(`📝 Logged ${type}: ${data.sessionId || 'unknown'}`);
  } catch (error) {
    console.error('Failed to log interaction:', error.message);
  }
}

// In-memory storage for analytics (in production, use a database)
const analyticsData = [];

// Enhanced analytics endpoint for data collection
app.post('/analytics', async (req, res) => {
  try {
    const { type, data } = req.body;
    
    const analyticsEntry = {
      id: Date.now().toString(),
      type,
      data,
      timestamp: new Date().toISOString(),
      ip: req.ip || req.connection.remoteAddress,
      userAgent: req.get('User-Agent'),
    };
    
    analyticsData.push(analyticsEntry);
    
    // Enhanced logging
    await logInteraction('analytics', {
      ...data,
      type,
      ip: analyticsEntry.ip,
      userAgent: analyticsEntry.userAgent.substring(0, 100)
    });
    
    res.json({ 
      success: true, 
      message: 'Analytics data collected',
      timestamp: analyticsEntry.timestamp
    });
  } catch (error) {
    console.error('Analytics error:', error);
    res.status(500).json({ error: 'Failed to collect analytics' });
  }
});

// Enhanced analytics dashboard endpoint
app.get('/analytics', async (req, res) => {
  try {
    const stats = {
      totalEntries: analyticsData.length,
      pageViews: analyticsData.filter(entry => entry.type === 'page_view').length,
      userInteractions: analyticsData.filter(entry => entry.type === 'user_interaction').length,
      uniqueUsers: new Set(analyticsData.map(entry => entry.data.sessionId)).size,
      recentEntries: analyticsData.slice(-10).reverse(),
      systemHealth: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
      }
    };
    
    res.json(stats);
  } catch (error) {
    console.error('Analytics stats error:', error);
    res.status(500).json({ error: 'Failed to get analytics' });
  }
});

// Enhanced AI endpoint with improved communication patterns
app.post('/ai', async (req, res) => {
  try {
    const { prompt, mood = 'neutral', trust = 0.5, persona = 'athenamyst' } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // Input validation and sanitization
    const trustLevel = Math.max(0, Math.min(1, parseFloat(trust) || 0.5));
    const sanitizedPrompt = prompt.substring(0, 1000); // Limit prompt length
    
    // Enhanced mock responses based on prompt content and mood
    let response = '';
    const lowerPrompt = sanitizedPrompt.toLowerCase();
    
    // Persona-based responses
    const personas = {
      athenamyst: {
        greeting: 'Hello! I\'m AthenaMyst, your AI assistant. How can I help you today?',
        weather: 'I can help you with weather information! However, this is a demo version. In the full version, I would connect to real weather APIs.',
        time: `The current time is ${new Date().toLocaleString()}. I can help you with time-related queries!`,
        help: 'I\'m here to help! You can ask me about various topics, and I\'ll do my best to assist you. This is a demo version showcasing our AI capabilities.',
        default: `AthenaMyst AI (public demo): You said "${sanitizedPrompt}". This is a demonstration of our AI interface. In the full version, I would provide more sophisticated responses and real AI processing.`
      },
      trader: {
        greeting: 'Greetings! I\'m AthenaMyst in trading mode. Ready to analyze market conditions.',
        weather: 'Market conditions are always changing. In the full version, I would provide real-time market analysis.',
        time: `Current market time: ${new Date().toLocaleString()}. Markets are dynamic and ever-evolving.`,
        help: 'I\'m here to assist with trading analysis. This demo shows our AI capabilities for market insights.',
        default: `Trading Analysis (demo): "${sanitizedPrompt}" - In the full version, I would provide real market analysis and trading insights.`
      }
    };
    
    const currentPersona = personas[persona] || personas.athenamyst;
    
    if (lowerPrompt.includes('hello') || lowerPrompt.includes('hi')) {
      response = currentPersona.greeting;
    } else if (lowerPrompt.includes('weather')) {
      response = currentPersona.weather;
    } else if (lowerPrompt.includes('time') || lowerPrompt.includes('date')) {
      response = currentPersona.time;
    } else if (lowerPrompt.includes('help')) {
      response = currentPersona.help;
    } else {
      response = currentPersona.default;
    }
    
    // Log the interaction
    await logInteraction('ai_interaction', {
      prompt: sanitizedPrompt,
      response: response.substring(0, 500),
      mood,
      trust: trustLevel,
      persona,
      sessionId: req.body.sessionId || 'unknown'
    });
    
    res.json({
      result: response,
      info: 'This is a public demo endpoint. Real AI processing would be available in the full version.',
      timestamp: new Date().toISOString(),
      persona,
      trust: trustLevel,
      mood
    });
  } catch (error) {
    console.error('AI endpoint error:', error);
    res.status(500).json({ error: 'Failed to process AI request' });
  }
});

// Enhanced health check endpoint
app.get('/health', (req, res) => {
  const healthData = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0',
    environment: process.env.NODE_ENV || 'development',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    endpoints: {
      ai: '/ai',
      analytics: '/analytics',
      health: '/health'
    }
  };
  
  res.json(healthData);
});

// Enhanced system status endpoint
app.get('/status', async (req, res) => {
  try {
    const status = {
      system: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        platform: process.platform,
        nodeVersion: process.version
      },
      analytics: {
        totalEntries: analyticsData.length,
        uniqueUsers: new Set(analyticsData.map(entry => entry.data.sessionId)).size,
        recentActivity: analyticsData.slice(-5).reverse()
      },
      logs: {
        interactionLog: interactionLogPath,
        analyticsLog: analyticsLogPath
      },
      timestamp: new Date().toISOString()
    };
    
    res.json(status);
  } catch (error) {
    console.error('Status endpoint error:', error);
    res.status(500).json({ error: 'Failed to get system status' });
  }
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/dist/index.html'));
});

// Enhanced error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
    path: req.path
  });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`🚀 AthenaMyst-Test API v2.0 running on port ${PORT}`);
  console.log(`📊 Analytics endpoint: http://localhost:${PORT}/analytics`);
  console.log(`🤖 AI endpoint: http://localhost:${PORT}/ai`);
  console.log(`💚 Health check: http://localhost:${PORT}/health`);
  console.log(`📈 System status: http://localhost:${PORT}/status`);
  console.log(`📝 Logs directory: ${logDir}`);
});