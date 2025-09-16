"""
Delta-Chat bot integration for DeepTreeEchoBot.
"""
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import deltachat
except ImportError:
    deltachat = None

from .config import DeltaChatConfig
from .message_handler import MessageHandler
from ..core.bot import DeepTreeEchoBot
from ..config.settings import BotConfig


class DeltaChatBot:
    """
    Delta-Chat integration wrapper for DeepTreeEchoBot.
    
    This class provides the bridge between Delta-Chat and the core DeepTreeEchoBot,
    handling message events, user interactions, and bot lifecycle.
    """
    
    def __init__(self, deep_tree_config: BotConfig, deltachat_config: DeltaChatConfig):
        self.deep_tree_config = deep_tree_config
        self.deltachat_config = deltachat_config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.deep_tree_bot: Optional[DeepTreeEchoBot] = None
        self.message_handler: Optional[MessageHandler] = None
        self.account: Optional[Any] = None  # deltachat.Account
        
        # Runtime state
        self.running = False
        self.startup_complete = False
        
        # Validate dependencies
        if not deltachat:
            raise ImportError(
                "deltachat package is required for Delta-Chat integration. "
                "Install with: pip install deltachat"
            )
            
    async def initialize(self) -> None:
        """Initialize the Delta-Chat bot."""
        try:
            self.logger.info("Initializing Delta-Chat bot...")
            
            # Validate configuration
            self.deltachat_config.validate()
            
            # Initialize core DeepTreeEchoBot
            self.deep_tree_bot = DeepTreeEchoBot(self.deep_tree_config)
            await self.deep_tree_bot.initialize()
            
            # Initialize message handler
            self.message_handler = MessageHandler(self.deep_tree_bot, self.deltachat_config)
            
            # Setup Delta-Chat account
            await self._setup_deltachat_account()
            
            self.startup_complete = True
            self.logger.info("Delta-Chat bot initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Delta-Chat bot: {e}")
            raise
            
    async def _setup_deltachat_account(self) -> None:
        """Setup and configure Delta-Chat account."""
        try:
            # Create database directory if needed
            db_path = Path(self.deltachat_config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create blob directory if needed
            blob_dir = Path(self.deltachat_config.blob_dir)
            blob_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Delta-Chat account
            self.account = deltachat.Account(str(db_path))
            
            # Set blob directory
            self.account.set_config("blobdir", str(blob_dir))
            
            # Configure display name
            self.account.set_config("displayname", self.deltachat_config.bot_name)
            
            # Set up email account
            self.account.set_config("addr", self.deltachat_config.email)
            self.account.set_config("mail_pw", self.deltachat_config.password)
            
            # Configure IMAP settings if provided
            if self.deltachat_config.imap_server:
                self.account.set_config("mail_server", self.deltachat_config.imap_server)
            if self.deltachat_config.imap_port:
                self.account.set_config("mail_port", str(self.deltachat_config.imap_port))
            if self.deltachat_config.imap_security:
                self.account.set_config("mail_security", self.deltachat_config.imap_security)
                
            # Configure SMTP settings if provided
            if self.deltachat_config.smtp_server:
                self.account.set_config("send_server", self.deltachat_config.smtp_server)
            if self.deltachat_config.smtp_port:
                self.account.set_config("send_port", str(self.deltachat_config.smtp_port))
            if self.deltachat_config.smtp_security:
                self.account.set_config("send_security", self.deltachat_config.smtp_security)
                
            # Set bot-specific configurations
            self.account.set_config("bot", "1")  # Mark as bot
            self.account.set_config("mdns_enabled", "0")  # Disable mDNS
            
            # Start account
            self.account.start_io()
            
            # Configure account - this will trigger the login process
            self.account.configure()
            
            # Wait for configuration to complete
            while not self.account.is_configured():
                await asyncio.sleep(0.1)
                
            self.logger.info(f"Delta-Chat account configured for {self.deltachat_config.email}")
            
            # Register event handlers
            self._register_event_handlers()
            
        except Exception as e:
            self.logger.error(f"Failed to setup Delta-Chat account: {e}")
            raise
            
    def _register_event_handlers(self) -> None:
        """Register Delta-Chat event handlers."""
        if not self.account:
            return
            
        # Register message handler
        @self.account.on(deltachat.events.NewMessage)
        def on_message(account, message):
            if self.message_handler:
                # Run async handler in event loop
                asyncio.create_task(self.message_handler.handle_message(account, message))
                
        # Register other useful events
        @self.account.on(deltachat.events.ContactRequest)
        def on_contact_request(account, contact):
            if self.deltachat_config.auto_accept_chats:
                self.logger.info(f"Auto-accepting contact request from {contact.addr}")
                # Contact requests are handled automatically by the message handler
                
        @self.account.on(deltachat.events.ConfigureProgress)
        def on_configure_progress(account, progress, comment):
            if progress == 1000:  # Configuration complete
                self.logger.info("Delta-Chat configuration complete")
            elif progress > 0:
                self.logger.debug(f"Configuration progress: {progress}/1000 - {comment}")
                
        self.logger.info("Delta-Chat event handlers registered")
        
    async def run(self) -> None:
        """Run the Delta-Chat bot."""
        if not self.startup_complete:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
            
        self.running = True
        self.logger.info("Starting Delta-Chat bot...")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Main event loop
            while self.running:
                # The actual message processing is handled by Delta-Chat events
                # We just need to keep the event loop running
                await asyncio.sleep(1)
                
                # Periodic maintenance
                if hasattr(self, '_last_maintenance'):
                    if (asyncio.get_event_loop().time() - self._last_maintenance) > 300:  # 5 minutes
                        await self._periodic_maintenance()
                else:
                    self._last_maintenance = asyncio.get_event_loop().time()
                    
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise
        finally:
            await self.shutdown()
            
    async def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        try:
            # Update timestamp
            self._last_maintenance = asyncio.get_event_loop().time()
            
            # Log some basic statistics
            if self.message_handler:
                stats = self.message_handler.get_stats()
                self.logger.debug(f"Bot stats: {stats}")
                
            # Clean up old rate limit data (older than 1 hour)
            if self.message_handler:
                import time
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour ago
                
                # Clean up old entries (simplified approach)
                to_remove = []
                for addr, last_time in self.message_handler.last_message_time.items():
                    if last_time.timestamp() < cutoff_time:
                        to_remove.append(addr)
                        
                for addr in to_remove:
                    self.message_handler.last_message_time.pop(addr, None)
                    self.message_handler.message_count.pop(addr, None)
                    
                if to_remove:
                    self.logger.debug(f"Cleaned up rate limit data for {len(to_remove)} users")
                    
        except Exception as e:
            self.logger.error(f"Error in periodic maintenance: {e}")
            
    async def shutdown(self) -> None:
        """Shutdown the Delta-Chat bot gracefully."""
        self.logger.info("Shutting down Delta-Chat bot...")
        self.running = False
        
        try:
            # Stop Delta-Chat account
            if self.account:
                self.account.stop_io()
                self.account = None
                
            # Shutdown core bot
            if self.deep_tree_bot:
                await self.deep_tree_bot.shutdown()
                
            self.logger.info("Delta-Chat bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = {
            'running': self.running,
            'startup_complete': self.startup_complete,
            'account_configured': self.account.is_configured() if self.account else False,
        }
        
        if self.message_handler:
            status.update(self.message_handler.get_stats())
            
        if self.deep_tree_bot:
            try:
                rl_stats = self.deep_tree_bot.rl_agent.get_training_stats()
                status['deep_tree_stats'] = rl_stats
            except:
                pass
                
        return status