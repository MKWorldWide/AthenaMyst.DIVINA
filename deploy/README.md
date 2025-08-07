# AthenaMyst DivineTrader Deployment Guide

This guide explains how to deploy the AthenaMyst DivineTrader with OANDA integration.

## Prerequisites

1. Node.js 16+ and npm installed
2. PM2 installed globally (`npm install -g pm2`)
3. OANDA API credentials (Account ID and API Key)
4. Environment variables configured (see `.env.example`)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Application
NODE_ENV=production
PORT=3000

# OANDA Configuration
OANDA_ACCOUNT_ID=your_account_id
OANDA_API_KEY=your_api_key
OANDA_ENV=practice  # or 'live' for production trading

# Deployment
DEPLOY_ENV=production
```

## Deployment Steps

### 1. Build the Application

```bash
npm run build
```

### 2. Run the Deployment Script

```bash
node deploy/oanda-deployment.js
```

This will:
1. Validate environment variables
2. Build the application
3. Copy necessary files to the deployment directory
4. Install production dependencies
5. Create a PM2 ecosystem file

### 3. Start the Application

Navigate to the deployment directory and start the application with PM2:

```bash
cd deploy/production
pm2 start ecosystem.config.js --env production
```

### 4. Verify the Deployment

Check the application logs:

```bash
pm2 logs
```

View the application status:

```bash
pm2 status
```

## Updating the Deployment

To update an existing deployment:

1. Pull the latest changes from the repository
2. Rebuild the application
3. Rerun the deployment script
4. Restart the PM2 process:

```bash
pm2 restart ecosystem.config.js --env production
```

## Monitoring

### PM2 Monitoring

```bash
# Monitor application
pm2 monit

# View logs in real-time
pm2 logs --lines 100

# View application status
pm2 status
```

### OANDA Account Monitoring

Regularly monitor your OANDA account through their web interface to track trades, account balance, and performance.

## Security Considerations

- Never commit sensitive information (API keys, account IDs) to version control
- Use environment variables for all sensitive configuration
- Regularly rotate API keys
- Monitor application logs for suspicious activity
- Set up proper firewall rules to restrict access to the trading server

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Ensure all required variables are set in the `.env` file
   - Verify the `.env` file has the correct permissions

2. **OANDA API Connection Issues**
   - Verify your API key is correct and has the necessary permissions
   - Check that your IP address is whitelisted in your OANDA account settings
   - Verify the OANDA environment (practice/live) matches your account type

3. **PM2 Process Not Starting**
   - Check the PM2 logs: `pm2 logs`
   - Verify the application builds successfully before deployment
   - Ensure all dependencies are installed in the deployment directory

For additional support, please refer to the project documentation or open an issue in the repository.
