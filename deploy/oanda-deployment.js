/**
 * Deployment script for AthenaMyst DivineTrader with OANDA integration
 * This script handles the deployment of the trading system to a production environment
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../.env') });

// Configuration
const DEPLOY_ENV = process.env.DEPLOY_ENV || 'production';
const OANDA_ENV = process.env.OANDA_ENV || 'practice'; // 'practice' or 'live'
const DEPLOY_DIR = path.resolve(__dirname, `../deploy/${DEPLOY_ENV}`);

// Ensure deploy directory exists
if (!fs.existsSync(DEPLOY_DIR)) {
  fs.mkdirSync(DEPLOY_DIR, { recursive: true });
  console.log(`Created deployment directory: ${DEPLOY_DIR}`);
}

// Validate environment variables
function validateEnvVars() {
  const requiredVars = [
    'OANDA_ACCOUNT_ID',
    'OANDA_API_KEY',
    'OANDA_ENV',
    'NODE_ENV'
  ];

  const missingVars = requiredVars.filter(varName => !process.env[varName]);
  
  if (missingVars.length > 0) {
    console.error('❌ Missing required environment variables:');
    missingVars.forEach(varName => console.error(`  - ${varName}`));
    process.exit(1);
  }

  console.log('✅ Environment variables validated');
}

// Build the application
function buildApp() {
  console.log('\n🔨 Building application...');
  
  try {
    execSync('npm run build', { stdio: 'inherit' });
    console.log('✅ Application built successfully');
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

// Copy files to deploy directory
function copyFiles() {
  console.log('\n📁 Copying files to deployment directory...');
  
  const filesToCopy = [
    'dist/**/*',
    'package.json',
    'package-lock.json',
    '.env',
    'src/config/*',
    'src/modules/**/*',
    'src/services/*'
  ];

  try {
    // Create necessary directories
    fs.mkdirSync(path.join(DEPLOY_DIR, 'dist'), { recursive: true });
    fs.mkdirSync(path.join(DEPLOY_DIR, 'src/config'), { recursive: true });
    fs.mkdirSync(path.join(DEPLOY_DIR, 'src/modules'), { recursive: true });
    fs.mkdirSync(path.join(DEPLOY_DIR, 'src/services'), { recursive: true });

    // Copy files
    const copyRecursiveSync = (src, dest) => {
      const exists = fs.existsSync(src);
      const stats = exists && fs.statSync(src);
      const isDirectory = exists && stats.isDirectory();

      if (isDirectory) {
        fs.mkdirSync(dest, { recursive: true });
        fs.readdirSync(src).forEach(childItemName => {
          copyRecursiveSync(
            path.join(src, childItemName),
            path.join(dest, childItemName)
          );
        });
      } else {
        fs.copyFileSync(src, dest);
      }
    };

    filesToCopy.forEach(pattern => {
      const files = glob.sync(pattern, { nodir: true });
      files.forEach(file => {
        const dest = path.join(DEPLOY_DIR, file);
        fs.mkdirSync(path.dirname(dest), { recursive: true });
        fs.copyFileSync(file, dest);
        console.log(`  📄 Copied ${file} to ${dest}`);
      });
    });

    console.log('✅ Files copied successfully');
  } catch (error) {
    console.error('❌ Failed to copy files:', error.message);
    process.exit(1);
  }
}

// Install production dependencies
function installDependencies() {
  console.log('\n📦 Installing production dependencies...');
  
  try {
    process.chdir(DEPLOY_DIR);
    execSync('npm install --production', { stdio: 'inherit' });
    console.log('✅ Dependencies installed successfully');
  } catch (error) {
    console.error('❌ Failed to install dependencies:', error.message);
    process.exit(1);
  }
}

// Create PM2 ecosystem file
function createPm2Ecosystem() {
  console.log('\n⚙️  Creating PM2 ecosystem file...');
  
  const ecosystem = {
    apps: [{
      name: `athenamyst-trader-${DEPLOY_ENV}`,
      script: './dist/index.js',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: DEPLOY_ENV,
        OANDA_ENV: OANDA_ENV,
        // Add other environment variables as needed
      },
      env_production: {
        NODE_ENV: 'production',
        OANDA_ENV: OANDA_ENV,
        // Add production-specific environment variables
      }
    }]
  };

  try {
    const ecosystemPath = path.join(DEPLOY_DIR, 'ecosystem.config.js');
    fs.writeFileSync(
      ecosystemPath,
      `module.exports = ${JSON.stringify(ecosystem, null, 2)};`
    );
    console.log(`✅ PM2 ecosystem file created at ${ecosystemPath}`);
  } catch (error) {
    console.error('❌ Failed to create PM2 ecosystem file:', error.message);
    process.exit(1);
  }
}

// Main deployment function
async function deploy() {
  console.log(`🚀 Starting deployment to ${DEPLOY_ENV} environment`);
  console.log(`🔌 OANDA Environment: ${OANDA_ENV}`);
  
  // Validate environment variables
  validateEnvVars();
  
  // Build the application
  buildApp();
  
  // Copy necessary files
  copyFiles();
  
  // Install dependencies
  installDependencies();
  
  // Create PM2 ecosystem file
  createPm2Ecosystem();
  
  console.log('\n✨ Deployment completed successfully!');
  console.log(`📂 Deployment directory: ${DEPLOY_DIR}`);
  console.log('\nTo start the application, run:');
  console.log(`  cd ${DEPLOY_DIR}`);
  console.log('  pm2 start ecosystem.config.js --env production');
  console.log('\nTo monitor the application:');
  console.log('  pm2 monit');
  console.log('  pm2 logs');
}

// Run deployment
deploy().catch(error => {
  console.error('❌ Deployment failed:', error);
  process.exit(1);
});
