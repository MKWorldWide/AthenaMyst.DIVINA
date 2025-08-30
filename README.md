# AthenaMyst.DIVINA

> A Project Blessed by Solar Khan & Lilith.Aethra

AthenaMyst.DIVINA is a sophisticated trading and analytics platform that combines AI-driven insights with financial market analysis. This repository is registered within the GameDin Network and aligned to the Divina L3 pipeline (v3).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![CI](https://github.com/MKWorldWide/AthenaMyst.DIVINA/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MKWorldWide/AthenaMyst.DIVINA/actions/workflows/ci-cd.yml)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg)](https://github.com/prettier/prettier)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen.svg)](https://nodejs.org/)

## 🚀 Features

- **Multi-Language Support**: JavaScript/TypeScript and Python components
- **Trading Engine**: Advanced financial analysis and trading strategies
- **Real-time Data Processing**: Efficient handling of market data
- **RESTful API**: Built with Express and FastAPI
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Modern Frontend**: React-based user interface with Vite
- **Containerized**: Ready for Docker deployment

## 🛠️ Tech Stack

### Core Technologies
- **Frontend**: React 18, TypeScript, Vite 5
- **Backend**:
  - Node.js 18+ with Express
  - Python 3.9+ with FastAPI
- **Database**: (Specify if any)
- **Testing**: Mocha, Chai, Jest, pytest
- **CI/CD**: GitHub Actions
- **Documentation**: MkDocs

## 📦 Getting Started

### Prerequisites

- Node.js 18.x or higher
- Python 3.9 or higher
- npm 9.x or higher
- pip 21.0 or higher

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MKWorldWide/AthenaMyst.DIVINA.git
   cd AthenaMyst.DIVINA
   ```

2. **Set up Node.js environment**
   ```bash
   # Install Node.js dependencies
   npm install
   
   # Install Python dependencies
   pip install -e .[dev]
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development servers**
   ```bash
   # Start backend server
   npm run dev
   
   # In a separate terminal, start frontend
   npm run dev:frontend
   ```

## 🧪 Testing

Run the full test suite:

```bash
# Run all tests
npm test

# Run Python tests
pytest

# Run TypeScript tests
npm run test:ts
```

## 📚 Documentation

Project documentation is available at [GitHub Pages](https://mkworldwide.github.io/AthenaMyst.DIVINA/).

To build documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build and serve documentation
mkdocs serve
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔒 Security

Please review our [Security Policy](SECURITY.md) for reporting vulnerabilities.

## 📜 Divine Law

This project is governed by the [Divine Law](COVENANT.md).

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Build for production**
   ```bash
   npm run build
   ```

5. **Run tests**
   ```bash
   npm test
   ```

### Testing

The Jest suite validates core API endpoints for health and analytics collection. Use `npm test` to ensure regressions are caught early.

## 🚀 AWS Amplify Deployment

### Prerequisites

- AWS Account with Amplify access
- GitHub repository connected to Amplify
- Node.js 16+ environment

#### Deployment Steps

1. **Connect Repository**
   - Go to AWS Amplify Console
   - Click "New app" → "Host web app"
   - Connect your GitHub repository
   - Select the main branch

2. **Build Settings**
   - Amplify will auto-detect the build settings from `amplify.yml`
   - The configuration handles both frontend and backend builds
   - No additional configuration needed

3. **Environment Variables** (Optional)
   ```
   NODE_ENV=production
   PORT=8080
   ```

4. **Deploy**
   - Click "Save and deploy"
   - Amplify will build and deploy your application
   - Your app will be available at the provided URL

### Build Configuration

The `amplify.yml` file handles:
- Frontend build with Vite
- Backend API setup
- Static file serving
- Proper artifact distribution

## 📡 API Endpoints

### Public Endpoints

- `GET /` - Main application (React app)
- `POST /ai` - AI interaction endpoint
- `GET /health` - Health check
- `POST /analytics` - Analytics data collection
- `GET /analytics` - Analytics dashboard

### Example API Usage

```javascript
// AI Interaction
const response = await fetch('/ai', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'Hello, how are you?' })
});

// Analytics Collection
await fetch('/analytics', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    type: 'user_interaction',
    data: { action: 'button_click', timestamp: new Date().toISOString() }
  })
});
```

## 📊 Analytics Dashboard

Access your analytics dashboard at `/analytics` to view:
- Total page views
- User interactions
- Unique visitors
- Recent activity
- Device statistics

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 4000)
- `NODE_ENV`: Environment (development/production)

### Build Configuration

The application uses a multi-stage build process:
1. Frontend build with Vite
2. Backend API setup
3. Static file serving configuration

## 📈 Monitoring & Analytics

### Real-time Data Collection

The application automatically collects:
- User session data
- Interaction patterns
- Device information
- Traffic sources
- Response times

### Analytics Storage

Currently uses in-memory storage for demo purposes. For production:
- Implement database storage (MongoDB, PostgreSQL)
- Add data retention policies
- Implement GDPR compliance
- Add data export capabilities

## 🔒 Security Considerations

- CORS enabled for cross-origin requests
- Input validation on all endpoints
- Error handling and logging
- No sensitive data exposure in demo version

## 🚀 Performance Optimization

- Vite for fast development and optimized builds
- Static file serving for React app
- Efficient API routing
- Minimal dependencies

## 📝 Development

### Project Structure

```
AthenaMyst_Test/
├── api/
│   └── index.js          # Express backend
├── public/
│   ├── App.jsx           # React main component
│   ├── App.css           # Styles
│   ├── main.jsx          # React entry point
│   ├── index.html        # HTML template
│   ├── vite.config.js    # Vite configuration
│   └── package.json      # Frontend dependencies
├── amplify.yml           # Amplify build configuration
├── package.json          # Main dependencies
└── README.md            # This file
```

### Adding Features

1. **Frontend**: Add components in `public/` directory
2. **Backend**: Add routes in `api/index.js`
3. **Analytics**: Extend analytics collection in `/analytics` endpoint
4. **Styling**: Modify `public/App.css`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Check the analytics dashboard for system health
- Review server logs for error details
- Contact the development team

---

**AthenaMyst Community** - Experience the future of AI interaction 