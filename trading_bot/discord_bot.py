import os
import discord
from discord.ext import commands, tasks
from discord import Webhook, AsyncWebhookAdapter
import aiohttp
import asyncio
import json
from dotenv import load_dotenv
from trading_engine import TradingEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DiscordBot')

# Load environment variables
load_dotenv()

class TradingBot(commands.Bot):
    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Initialize the bot
        super().__init__(command_prefix='!', intents=intents)
        
        # Initialize trading engine
        self.trading_engine = TradingEngine()
        
        # Webhook URL for sending notifications
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.alert_role_id = os.getenv('DISCORD_ALERT_ROLE_ID')
        
        # Background task to run the trading strategy
        self.trading_task.start()
    
    async def setup_hook(self):
        """Set up the bot's commands and background tasks."""
        await self.add_cog(TradingCommands(self))
        logger.info("Bot setup complete")
    
    async def on_ready(self):
        """Event that runs when the bot is ready."""
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        logger.info('------')
    
    async def send_webhook(self, title: str, description: str, color: int = 0x3498db, fields: list = None):
        """Send a message to the configured webhook."""
        if not self.webhook_url:
            logger.warning("No webhook URL configured")
            return
        
        # Create embed
        embed = discord.Embed(
            title=title,
            description=description,
            color=color
        )
        
        # Add fields if provided
        if fields:
            for name, value, inline in fields:
                embed.add_field(name=name, value=value, inline=inline)
        
        # Add timestamp
        embed.timestamp = discord.utils.utcnow()
        
        # Send the webhook
        async with aiohttp.ClientSession() as session:
            webhook = Webhook.from_url(self.webhook_url, session=session, adapter=AsyncWebhookAdapter(session))
            try:
                content = f"<@&{self.alert_role_id}>" if self.alert_role_id else None
                await webhook.send(content=content, embed=embed, username=f"Trading Bot - {self.user.name}")
                logger.info("Sent webhook notification")
            except Exception as e:
                logger.error(f"Error sending webhook: {e}")
    
    @tasks.loop(minutes=15)  # Run every 15 minutes
    async def trading_task(self):
        """Background task to run the trading strategy."""
        try:
            logger.info("Running trading strategy...")
            signal = await self.trading_engine.run_strategy()
            
            if signal and signal.get('action') != 'HOLD':
                # Format signal for display
                color = 0x00ff00 if signal['action'] == 'BUY' else 0xff0000
                action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
                
                fields = [
                    ("Action", f"{action_emoji} {signal['action']} {signal['pair']}", True),
                    ("Price", f"${signal['price']:,.2f}", True),
                    ("Stop Loss", f"${signal['stop_loss']:,.2f}", True) if signal['stop_loss'] else ("", "", True),
                    ("Take Profit", f"${signal['take_profit']:,.2f}", True) if signal['take_profit'] else ("", "", True),
                    ("RSI", f"{signal['indicators']['rsi']:.2f}", True),
                    ("MACD", f"{signal['indicators']['macd']:.4f}", True),
                    ("Signal", f"{signal['indicators']['macd_signal']:.4f}", True),
                    ("ATR", f"{signal['indicators']['atr']:.4f}", True)
                ]
                
                # Send webhook notification
                await self.send_webhook(
                    title=f"New Trade Signal: {signal['pair']}",
                    description=f"Automated trading signal generated",
                    color=color,
                    fields=fields
                )
                
        except Exception as e:
            logger.error(f"Error in trading task: {e}")
            await self.send_webhook(
                title="⚠️ Trading Error",
                description=f"An error occurred in the trading task: ```{str(e)}```",
                color=0xff0000
            )
    
    @trading_task.before_loop
    async def before_trading_task(self):
        """Wait until the bot is ready before starting the trading task."""
        await self.wait_until_ready()

class TradingCommands(commands.Cog):
    """Commands for interacting with the trading bot."""
    
    def __init__(self, bot):
        self.bot = bot
        self.trading_engine = bot.trading_engine
    
    @commands.command(name='forecast')
    async def forecast(self, ctx, pair: str = None):
        """Get the current market forecast for a trading pair."""
        try:
            # Update trading pair if provided
            if pair:
                self.trading_engine.trading_pair = pair.upper().replace('-', '/')
            
            # Get the latest signal
            signal = await self.trading_engine.run_strategy()
            
            if not signal:
                await ctx.send("❌ Failed to generate forecast. Please try again later.")
                return
            
            # Determine color and emoji based on signal
            if signal['action'] == 'BUY':
                color = 0x00ff00
                action_emoji = "🟢"
            elif signal['action'] == 'SELL':
                color = 0xff0000
                action_emoji = "🔴"
            else:
                color = 0x3498db
                action_emoji = "🟡"
            
            # Create embed
            embed = discord.Embed(
                title=f"📊 {signal['pair']} Market Forecast",
                description=f"**Signal:** {action_emoji} {signal['action']}",
                color=color,
                timestamp=discord.utils.utcnow()
            )
            
            # Add price information
            embed.add_field(name="Current Price", value=f"${signal['price']:,.2f}", inline=True)
            
            # Add indicators
            embed.add_field(name="RSI", value=f"{signal['indicators']['rsi']:.2f}", inline=True)
            embed.add_field(name="EMA Fast", value=f"{signal['indicators']['ema_fast']:.2f}", inline=True)
            embed.add_field(name="EMA Slow", value=f"{signal['indicators']['ema_slow']:.2f}", inline=True)
            embed.add_field(name="MACD", value=f"{signal['indicators']['macd']:.4f}", inline=True)
            embed.add_field(name="Signal", value=f"{signal['indicators']['macd_signal']:.4f}", inline=True)
            embed.add_field(name="ATR", value=f"{signal['indicators']['atr']:.4f}", inline=True)
            
            # Add trading levels if available
            if signal['stop_loss'] and signal['take_profit']:
                embed.add_field(
                    name="Trading Levels",
                    value=f"🛑 Stop Loss: ${signal['stop_loss']:,.2f}\n🎯 Take Profit: ${signal['take_profit']:,.2f}",
                    inline=False
                )
            
            # Set footer with timestamp
            embed.set_footer(text="Trading Bot • Market data")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in forecast command: {e}")
            await ctx.send(f"❌ An error occurred: {str(e)}")
    
    @commands.command(name='balance')
    async def balance(self, ctx):
        """Check the current account balance."""
        try:
            balance = await self.trading_engine.exchange.fetch_balance()
            if not balance:
                await ctx.send("❌ Failed to fetch balance.")
                return
            
            # Get relevant balances
            free_balance = {k: v for k, v in balance['free'].items() if v > 0}
            used_balance = {k: v for k, v in balance['used'].items() if v > 0}
            total_balance = {k: v for k, v in balance['total'].items() if v > 0}
            
            # Create embed
            embed = discord.Embed(
                title="💰 Account Balance",
                color=0x3498db,
                timestamp=discord.utils.utcnow()
            )
            
            # Add balance information
            if free_balance:
                free_text = "\n".join([f"{k}: {v:.8f}" for k, v in free_balance.items()])
                embed.add_field(name="Free", value=f"```{free_text}```", inline=False)
            
            if used_balance:
                used_text = "\n".join([f"{k}: {v:.8f}" for k, v in used_balance.items()])
                embed.add_field(name="In Orders", value=f"```{used_text}```", inline=False)
            
            if total_balance:
                total_text = "\n".join([f"{k}: {v:.8f}" for k, v in total_balance.items()])
                embed.add_field(name="Total", value=f"```{total_text}```", inline=False)
            
            embed.set_footer(text="Trading Bot • Account")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            await ctx.send(f"❌ An error occurred: {str(e)}")
    
    @commands.command(name='positions')
    async def positions(self, ctx):
        """Check open positions."""
        try:
            positions = await self.trading_engine.exchange.fetch_positions()
            if not positions:
                await ctx.send("No open positions.")
                return
            
            # Filter out zero positions
            open_positions = [p for p in positions if float(p['contracts']) != 0]
            
            if not open_positions:
                await ctx.send("No open positions.")
                return
            
            # Create embed
            embed = discord.Embed(
                title="📊 Open Positions",
                color=0x3498db,
                timestamp=discord.utils.utcnow()
            )
            
            for pos in open_positions:
                symbol = pos['symbol']
                side = pos['side'].capitalize()
                contracts = float(pos['contracts'])
                entry_price = float(pos['entryPrice'])
                mark_price = float(pos['markPrice'])
                pnl = float(pos['unrealizedPnl'])
                pnl_pct = (pnl / (entry_price * abs(contracts))) * 100 if entry_price > 0 else 0
                
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                
                embed.add_field(
                    name=f"{symbol} - {side}",
                    value=(
                        f"**Contracts:** {contracts:.4f}\n"
                        f"**Entry:** ${entry_price:,.2f}\n"
                        f"**Mark Price:** ${mark_price:,.2f}\n"
                        f"**P&L:** {pnl_emoji} ${abs(pnl):.2f} ({abs(pnl_pct):.2f}%)"
                    ),
                    inline=True
                )
            
            embed.set_footer(text="Trading Bot • Positions")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await ctx.send(f"❌ An error occurred: {str(e)}")

# Run the bot
if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv('DISCORD_BOT_TOKEN'):
        logger.error("DISCORD_BOT_TOKEN environment variable is not set")
        exit(1)
    
    # Create and run the bot
    bot = TradingBot()
    bot.run(os.getenv('DISCORD_BOT_TOKEN'))
