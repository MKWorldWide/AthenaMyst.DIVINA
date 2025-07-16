const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public/dist')));

// In-memory storage for analytics (in production, use a database)
const analyticsData = [];

// Analytics endpoint for data collection
app.post('/analytics', (req, res) => {
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
    
    // Log analytics data (in production, save to database)
    console.log('Analytics collected:', {
      type,
      timestamp: analyticsEntry.timestamp,
      userAgent: analyticsEntry.userAgent.substring(0, 100) + '...',
    });
    
    res.json({ success: true, message: 'Analytics data collected' });
  } catch (error) {
    console.error('Analytics error:', error);
    res.status(500).json({ error: 'Failed to collect analytics' });
  }
});

// Analytics dashboard endpoint (for monitoring)
app.get('/analytics', (req, res) => {
  try {
    const stats = {
      totalEntries: analyticsData.length,
      pageViews: analyticsData.filter(entry => entry.type === 'page_view').length,
      userInteractions: analyticsData.filter(entry => entry.type === 'user_interaction').length,
      uniqueUsers: new Set(analyticsData.map(entry => entry.data.sessionId)).size,
      recentEntries: analyticsData.slice(-10).reverse(),
    };
    
    res.json(stats);
  } catch (error) {
    console.error('Analytics stats error:', error);
    res.status(500).json({ error: 'Failed to get analytics' });
  }
});

// Mock AI endpoint with enhanced responses
app.post('/ai', (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // Enhanced mock responses based on prompt content
    let response = '';
    const lowerPrompt = prompt.toLowerCase();
    
    if (lowerPrompt.includes('hello') || lowerPrompt.includes('hi')) {
      response = 'Hello! I\'m AthenaMyst, your AI assistant. How can I help you today?';
    } else if (lowerPrompt.includes('weather')) {
      response = 'I can help you with weather information! However, this is a demo version. In the full version, I would connect to real weather APIs.';
    } else if (lowerPrompt.includes('time') || lowerPrompt.includes('date')) {
      response = `The current time is ${new Date().toLocaleString()}. I can help you with time-related queries!`;
    } else if (lowerPrompt.includes('help')) {
      response = 'I\'m here to help! You can ask me about various topics, and I\'ll do my best to assist you. This is a demo version showcasing our AI capabilities.';
    } else {
      response = `AthenaMyst AI (public demo): You said "${prompt}". This is a demonstration of our AI interface. In the full version, I would provide more sophisticated responses and real AI processing.`;
    }
    
    res.json({
      result: response,
      info: 'This is a public demo endpoint. Real AI processing would be available in the full version.',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('AI endpoint error:', error);
    res.status(500).json({ error: 'Failed to process AI request' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  });
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/dist/index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`🚀 AthenaMyst-Test API running on port ${PORT}`);
  console.log(`📊 Analytics endpoint: http://localhost:${PORT}/analytics`);
  console.log(`🤖 AI endpoint: http://localhost:${PORT}/ai`);
  console.log(`💚 Health check: http://localhost:${PORT}/health`);
});