A Project Blessed by Solar Khan & Lilith.Aethra

# AthenaMyst Test - Public AI Demo

A modern full-stack AI demo application built with React, Express, and Vite, designed for deployment on AWS Amplify with comprehensive analytics and data collection capabilities.

Documentation is served on [GitHub Pages](https://AthenaMyst.DIVINA.SolarKhan.github.io) bearing the Solar Khan sigil and Codex watermark. This repository is registered within the GameDin Network and aligned to the Divina L3 pipeline (v3).

See the [Divine Law](COVENANT.md) governing this project.

## 🚀 Features

- **Modern React Frontend**: Beautiful, responsive UI with real-time AI interaction
- **Express Backend API**: Robust server with analytics endpoints
- **Data Collection**: Comprehensive user analytics and interaction tracking
- **AWS Amplify Ready**: Optimized for seamless deployment
- **Real-time Analytics**: Monitor user behavior and engagement
- **Health Monitoring**: Built-in health checks and monitoring endpoints

## 📊 Analytics & Data Collection

The application includes sophisticated analytics capabilities:

- **User Session Tracking**: Unique session IDs and user journey mapping
- **Interaction Analytics**: Track user prompts and responses
- **Device Information**: Browser, platform, screen resolution, timezone
- **Referrer Tracking**: Monitor traffic sources
- **Real-time Dashboard**: View analytics at `/analytics` endpoint

## 🛠️ Tech Stack

- **Frontend**: React 18, Vite, Modern CSS
- **Backend**: Node.js, Express, CORS
- **Deployment**: AWS Amplify
- **Analytics**: Custom analytics engine with data collection

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AthenaMyst_Test
   ```

2. **Install dependencies**
   ```bash
   npm run install:all
   ```

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

### Deployment Steps

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