const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Mock AI endpoint
app.post('/ai', (req, res) => {
  const { prompt } = req.body;
  res.json({
    result: `AthenaMyst AI (public demo): You said "${prompt}"`,
    info: 'This is a public mock endpoint. No real AI or trading logic is exposed.'
  });
});

// Health check
app.get('/', (req, res) => {
  res.send('AthenaMyst-Test public API is running.');
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`AthenaMyst-Test API running on port ${PORT}`);
}); 