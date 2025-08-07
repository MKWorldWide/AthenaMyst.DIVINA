import { EventEmitter } from 'events';
import { RiskManager } from './riskManager';
import { TechnicalIndicators } from './indicators';

type AlertLevel = 'info' | 'warning' | 'error' | 'critical';

interface Alert {
  id: string;
  timestamp: Date;
  level: AlertLevel;
  message: string;
  data?: Record<string, any>;
  acknowledged: boolean;
  acknowledgedAt?: Date;
  acknowledgedBy?: string;
}

interface MonitorConfig {
  checkInterval: number; // in milliseconds
  maxQueueSize: number;
  alertRetentionDays: number;
  emailNotifications?: {
    enabled: boolean;
    recipients: string[];
    minAlertLevel: AlertLevel;
  };
  slackWebhookUrl?: string;
}

export class TradingMonitor extends EventEmitter {
  private alerts: Map<string, Alert> = new Map();
  private metrics: Record<string, any> = {};
  private checkInterval: NodeJS.Timeout | null = null;
  private config: MonitorConfig;
  private riskManager: RiskManager;

  constructor(riskManager: RiskManager, config: Partial<MonitorConfig> = {}) {
    super();
    this.riskManager = riskManager;
    
    // Default configuration
    this.config = {
      checkInterval: 60000, // 1 minute
      maxQueueSize: 1000,
      alertRetentionDays: 7,
      ...config
    };
    
    // Set up event listeners
    this.setupEventListeners();
  }

  /**
   * Start monitoring the trading system
   */
  start(): void {
    if (this.checkInterval) {
      this.stop();
    }
    
    // Initial check
    this.performChecks();
    
    // Set up periodic checks
    this.checkInterval = setInterval(
      () => this.performChecks(),
      this.config.checkInterval
    );
    
    this.emit('monitoring:started');
  }

  /**
   * Stop monitoring the trading system
   */
  stop(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    this.emit('monitoring:stopped');
  }

  /**
   * Perform system health checks
   */
  private performChecks(): void {
    try {
      this.checkSystemHealth();
      this.checkRiskMetrics();
      this.checkOpenTrades();
      this.checkConnectionHealth();
      this.cleanupOldAlerts();
    } catch (error) {
      this.emit('error', error);
    }
  }

  /**
   * Check overall system health
   */
  private checkSystemHealth(): void {
    // Check memory usage
    const memoryUsage = process.memoryUsage();
    const memoryUsageMB = {
      rss: this.bytesToMB(memoryUsage.rss),
      heapTotal: this.bytesToMB(memoryUsage.heapTotal),
      heapUsed: this.bytesToMB(memoryUsage.heapUsed),
      external: this.bytesToMB(memoryUsage.external || 0),
      arrayBuffers: this.bytesToMB(memoryUsage.arrayBuffers || 0)
    };

    this.metrics.memory = memoryUsageMB;

    // Warn if memory usage is high
    if (memoryUsage.heapUsed / memoryUsage.heapTotal > 0.8) {
      this.raiseAlert(
        'high_memory_usage',
        'warning',
        'High memory usage detected',
        { memoryUsage: memoryUsageMB }
      );
    }

    // Check event loop lag
    const start = process.hrtime();
    setImmediate(() => {
      const delta = process.hrtime(start);
      const lagMs = delta[0] * 1000 + delta[1] / 1e6;
      this.metrics.eventLoopLag = lagMs;

      if (lagMs > 200) { // 200ms threshold
        this.raiseAlert(
          'high_event_loop_lag',
          'warning',
          `High event loop lag detected: ${lagMs.toFixed(2)}ms`,
          { lagMs }
        );
      }
    });
  }

  /**
   * Check risk metrics and raise alerts if needed
   */
  private checkRiskMetrics(): void {
    const metrics = this.riskManager.getRiskMetrics();
    this.metrics.risk = metrics;

    // Check daily drawdown
    if (metrics.currentDrawdownPct >= metrics.maxDailyDrawdown * 0.8) {
      this.raiseAlert(
        'high_daily_drawdown',
        metrics.currentDrawdownPct >= metrics.maxDailyDrawdown ? 'critical' : 'warning',
        `Daily drawdown at ${metrics.currentDrawdownPct.toFixed(2)}% ` +
        `(limit: ${metrics.maxDailyDrawdown}%)`,
        { metrics }
      );
    }

    // Check position concentration
    if (metrics.totalPositionValuePct > 30) {
      this.raiseAlert(
        'high_position_concentration',
        'warning',
        `High position concentration: ${metrics.totalPositionValuePct.toFixed(2)}% of account`,
        { metrics }
      );
    }
  }

  /**
   * Check status of open trades
   */
  private checkOpenTrades(): void {
    const openTrades = this.riskManager.getOpenTrades();
    this.metrics.openTrades = openTrades.length;

    // Check for stale trades (open too long)
    const now = new Date();
    const STALE_TRADE_HOURS = 24;
    
    openTrades.forEach(trade => {
      const tradeAgeHours = (now.getTime() - new Date(trade.openTime).getTime()) / (1000 * 60 * 60);
      
      if (tradeAgeHours > STALE_TRADE_HOURS) {
        this.raiseAlert(
          `stale_trade_${trade.id}`,
          'warning',
          `Trade ${trade.id} has been open for ${tradeAgeHours.toFixed(1)} hours`,
          { tradeId: trade.id, pair: trade.pair, ageHours: tradeAgeHours }
        );
      }
    });
  }

  /**
   * Check connection health to external services
   */
  private async checkConnectionHealth(): Promise<void> {
    // This would be implemented to check connections to:
    // - OANDA API
    // - Database
    // - Any other external services
    
    // Example implementation would go here
    this.metrics.lastConnectionCheck = new Date().toISOString();
  }

  /**
   * Clean up old alerts
   */
  private cleanupOldAlerts(): void {
    const retentionTime = this.config.alertRetentionDays * 24 * 60 * 60 * 1000; // days to ms
    const now = new Date();
    
    for (const [id, alert] of this.alerts.entries()) {
      const alertAge = now.getTime() - alert.timestamp.getTime();
      if (alertAge > retentionTime) {
        this.alerts.delete(id);
      }
    }
  }

  /**
   * Raise a new alert
   */
  raiseAlert(
    id: string,
    level: AlertLevel,
    message: string,
    data?: Record<string, any>
  ): Alert {
    const existingAlert = this.alerts.get(id);
    
    // If this is a duplicate alert, update the existing one
    if (existingAlert) {
      const updatedAlert: Alert = {
        ...existingAlert,
        message,
        data: { ...existingAlert.data, ...data },
        timestamp: new Date()
      };
      
      this.alerts.set(id, updatedAlert);
      this.emit('alert:updated', updatedAlert);
      return updatedAlert;
    }
    
    // Create a new alert
    const newAlert: Alert = {
      id,
      timestamp: new Date(),
      level,
      message,
      data,
      acknowledged: false
    };
    
    // Enforce max queue size
    if (this.alerts.size >= this.config.maxQueueSize) {
      // Remove the oldest alert
      const oldestAlert = Array.from(this.alerts.values())
        .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())[0];
      this.alerts.delete(oldestAlert.id);
    }
    
    this.alerts.set(id, newAlert);
    this.emit('alert:raised', newAlert);
    
    // Trigger notifications if configured
    this.handleAlertNotification(newAlert);
    
    return newAlert;
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string, acknowledgedBy: string): boolean {
    const alert = this.alerts.get(alertId);
    
    if (!alert) {
      return false;
    }
    
    const updatedAlert: Alert = {
      ...alert,
      acknowledged: true,
      acknowledgedAt: new Date(),
      acknowledgedBy
    };
    
    this.alerts.set(alertId, updatedAlert);
    this.emit('alert:acknowledged', updatedAlert);
    
    return true;
  }

  /**
   * Get all alerts, optionally filtered
   */
  getAlerts(filter?: {
    level?: AlertLevel | AlertLevel[];
    acknowledged?: boolean;
    startDate?: Date;
    endDate?: Date;
  }): Alert[] {
    let alerts = Array.from(this.alerts.values());
    
    if (filter) {
      if (filter.level) {
        const levels = Array.isArray(filter.level) ? filter.level : [filter.level];
        alerts = alerts.filter(alert => levels.includes(alert.level));
      }
      
      if (filter.acknowledged !== undefined) {
        alerts = alerts.filter(alert => alert.acknowledged === filter.acknowledged);
      }
      
      if (filter.startDate) {
        alerts = alerts.filter(alert => alert.timestamp >= filter.startDate!);
      }
      
      if (filter.endDate) {
        alerts = alerts.filter(alert => alert.timestamp <= filter.endDate!);
      }
    }
    
    return alerts.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  /**
   * Get current system metrics
   */
  getMetrics(): Record<string, any> {
    return { ...this.metrics };
  }

  /**
   * Handle alert notifications (email, Slack, etc.)
   */
  private handleAlertNotification(alert: Alert): void {
    // Skip if alert level is below minimum for notifications
    const minLevel = this.config.emailNotifications?.minAlertLevel || 'warning';
    const levelPriority = { info: 1, warning: 2, error: 3, critical: 4 };
    
    if (levelPriority[alert.level] < levelPriority[minLevel]) {
      return;
    }
    
    // Send email notification if enabled
    if (this.config.emailNotifications?.enabled) {
      this.sendEmailNotification(alert);
    }
    
    // Send Slack notification if webhook is configured
    if (this.config.slackWebhookUrl) {
      this.sendSlackNotification(alert);
    }
  }

  /**
   * Send email notification for an alert
   */
  private async sendEmailNotification(alert: Alert): Promise<void> {
    // Implementation would use Nodemailer or a similar library
    // This is a placeholder for the actual implementation
    console.log(`[Email] ${alert.level.toUpperCase()}: ${alert.message}`);
    
    // Example implementation:
    /*
    const transporter = nodemailer.createTransport({...});
    await transporter.sendMail({
      from: 'trading-bot@example.com',
      to: this.config.emailNotifications.recipients.join(','),
      subject: `[${alert.level.toUpperCase()}] ${alert.message}`,
      text: `Alert: ${alert.message}\n\n` +
            `Level: ${alert.level}\n` +
            `Time: ${alert.timestamp.toISOString()}\n` +
            `Details: ${JSON.stringify(alert.data, null, 2)}`
    });
    */
  }

  /**
   * Send Slack notification for an alert
   */
  private async sendSlackNotification(alert: Alert): Promise<void> {
    // Implementation would use the Slack webhook
    // This is a placeholder for the actual implementation
    console.log(`[Slack] ${alert.level.toUpperCase()}: ${alert.message}`);
    
    // Example implementation:
    /*
    const message = {
      text: `*${alert.level.toUpperCase()}*: ${alert.message}`,
      attachments: [
        {
          color: this.getAlertColor(alert.level),
          fields: [
            { title: 'Time', value: alert.timestamp.toISOString(), short: true },
            { title: 'Level', value: alert.level, short: true },
            { 
              title: 'Details', 
              value: '```' + JSON.stringify(alert.data, null, 2) + '```',
              short: false
            }
          ]
        }
      ]
    };
    
    await axios.post(this.config.slackWebhookUrl, message);
    */
  }

  /**
   * Get color for alert level (for Slack/UI)
   */
  private getAlertColor(level: AlertLevel): string {
    switch (level) {
      case 'info': return '#3498db';    // Blue
      case 'warning': return '#f39c12'; // Orange
      case 'error': return '#e74c3c';   // Red
      case 'critical': return '#8e44ad'; // Purple
      default: return '#95a5a6';        // Gray
    }
  }

  /**
   * Set up event listeners
   */
  private setupEventListeners(): void {
    // Listen for trade events
    this.riskManager.on('trade:opened', (trade: any) => {
      this.raiseAlert(
        `trade_opened_${trade.id}`,
        'info',
        `New trade opened: ${trade.pair} ${trade.direction} @ ${trade.entryPrice}`,
        { trade }
      );
    });
    
    this.riskManager.on('trade:closed', (trade: any) => {
      this.raiseAlert(
        `trade_closed_${trade.id}`,
        'info',
        `Trade closed: ${trade.pair} ${trade.direction} | P&L: ${trade.pnl} (${trade.pnlPct}%)`,
        { trade }
      );
    });
    
    // Listen for risk events
    this.riskManager.on('risk:limit_reached', (data: any) => {
      this.raiseAlert(
        `risk_limit_${data.type}`,
        'warning',
        `Risk limit reached: ${data.message}`,
        data
      );
    });
  }

  /**
   * Convert bytes to megabytes
   */
  private bytesToMB(bytes: number): number {
    return Math.round((bytes / 1024 / 1024) * 100) / 100;
  }
}

export default TradingMonitor;
