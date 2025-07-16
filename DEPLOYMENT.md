# 🚀 AWS Amplify Deployment Guide

This guide will walk you through deploying your AthenaMyst Test application to AWS Amplify for public access and data collection.

## 📋 Prerequisites

- AWS Account with Amplify access
- GitHub repository with your code
- Node.js 16+ knowledge
- Basic understanding of AWS services

## 🔧 Step 1: Prepare Your Repository

### 1.1 Verify Project Structure
Ensure your repository has the following structure:
```
AthenaMyst_Test/
├── api/
│   └── index.js
├── public/
│   ├── App.jsx
│   ├── App.css
│   ├── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── amplify.yml
├── package.json
├── README.md
└── .gitignore
```

### 1.2 Commit and Push
```bash
git add .
git commit -m "Prepare for Amplify deployment"
git push origin main
```

## 🌐 Step 2: AWS Amplify Setup

### 2.1 Access Amplify Console
1. Go to [AWS Amplify Console](https://console.aws.amazon.com/amplify/)
2. Sign in with your AWS account
3. Click "New app" → "Host web app"

### 2.2 Connect Repository
1. Choose "GitHub" as your repository service
2. Authorize AWS Amplify to access your GitHub account
3. Select your `AthenaMyst_Test` repository
4. Choose the `main` branch

### 2.3 Configure Build Settings
Amplify will auto-detect the build configuration from `amplify.yml`. Verify these settings:

**Build settings:**
- Build image: Amazon Linux:2023
- Node.js version: 18.x
- Build commands: Auto-detected from `amplify.yml`

**Environment variables (optional):**
```
NODE_ENV=production
PORT=8080
```

### 2.4 Deploy
1. Click "Save and deploy"
2. Monitor the build process in the Amplify console
3. Wait for deployment to complete (usually 5-10 minutes)

## 📊 Step 3: Verify Deployment

### 3.1 Check Application
1. Visit your Amplify app URL (provided after deployment)
2. Test the AI interaction feature
3. Verify the UI loads correctly

### 3.2 Test Analytics
1. Visit `https://your-app-url/analytics`
2. Check that analytics data is being collected
3. Verify the dashboard shows user interactions

### 3.3 Test API Endpoints
```bash
# Health check
curl https://your-app-url/health

# AI endpoint
curl -X POST https://your-app-url/ai \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, how are you?"}'

# Analytics endpoint
curl https://your-app-url/analytics
```

## 🔍 Step 4: Monitor and Optimize

### 4.1 Amplify Console Monitoring
- **Build History**: Monitor build success/failure
- **Performance**: Check app performance metrics
- **Access Logs**: View user access patterns
- **Error Tracking**: Monitor application errors

### 4.2 Analytics Dashboard
Access your analytics at `/analytics` to monitor:
- User sessions and interactions
- Device and browser statistics
- Traffic patterns and sources
- Response times and performance

### 4.3 Custom Domain (Optional)
1. Go to Amplify Console → Your App → Domain Management
2. Add your custom domain
3. Configure DNS settings
4. Enable SSL certificate

## 🛠️ Step 5: Advanced Configuration

### 5.1 Environment Variables
Add these in Amplify Console → App Settings → Environment Variables:
```
NODE_ENV=production
PORT=8080
ANALYTICS_ENABLED=true
```

### 5.2 Build Optimizations
The `amplify.yml` file is optimized for:
- Fast frontend builds with Vite
- Efficient backend deployment
- Proper artifact distribution
- Caching for faster subsequent builds

### 5.3 Performance Monitoring
- Enable AWS CloudWatch for detailed monitoring
- Set up alerts for performance issues
- Monitor API response times
- Track user engagement metrics

## 🔒 Step 6: Security Considerations

### 6.1 Data Collection Compliance
- Ensure GDPR compliance for EU users
- Implement data retention policies
- Add privacy policy and terms of service
- Consider data anonymization

### 6.2 API Security
- Rate limiting (implement in production)
- Input validation (already implemented)
- CORS configuration (already configured)
- Error handling (already implemented)

## 📈 Step 7: Data Collection Strategy

### 7.1 Current Analytics
The application collects:
- User session data
- Interaction patterns
- Device information
- Traffic sources
- Response times

### 7.2 Production Enhancements
For production deployment, consider:
- Database integration (MongoDB, PostgreSQL)
- Real-time analytics dashboard
- Data export capabilities
- Advanced user segmentation

## 🚨 Troubleshooting

### Common Issues

**Build Failures:**
- Check Node.js version compatibility
- Verify all dependencies are in package.json
- Review build logs in Amplify console

**API Errors:**
- Check CORS configuration
- Verify endpoint URLs
- Review server logs

**Analytics Issues:**
- Check browser console for errors
- Verify analytics endpoint is accessible
- Review data collection logic

### Support Resources
- [AWS Amplify Documentation](https://docs.aws.amazon.com/amplify/)
- [Amplify Console Guide](https://docs.aws.amazon.com/amplify/latest/userguide/welcome.html)
- [GitHub Issues](https://github.com/your-repo/issues)

## 🎯 Success Metrics

After deployment, monitor these key metrics:
- **Uptime**: 99.9%+ availability
- **Response Time**: < 2 seconds for API calls
- **User Engagement**: Track session duration and interactions
- **Data Collection**: Monitor analytics data volume
- **Error Rate**: < 1% error rate

## 📞 Next Steps

1. **Monitor Performance**: Use Amplify console and analytics dashboard
2. **Scale as Needed**: Amplify auto-scales based on traffic
3. **Enhance Features**: Add more AI capabilities and analytics
4. **User Feedback**: Collect user feedback and iterate
5. **Security Audits**: Regular security reviews and updates

---

**Your AthenaMyst Test application is now live and collecting data!** 🎉

For questions or issues, refer to the troubleshooting section or contact the development team. 