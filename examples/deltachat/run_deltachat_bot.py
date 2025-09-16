#!/usr/bin/env python3
"""
Example script for running DeepTreeEchoBot with Delta-Chat integration.

This example demonstrates how to set up and run the bot programmatically
instead of using the CLI.
"""
import asyncio
import logging
import os
from pathlib import Path

from deep_tree_echo_bot import BotConfig
from deep_tree_echo_bot.deltachat_integration import DeltaChatBot, DeltaChatConfig


async def main():
    """Main function to run the Delta-Chat bot."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create DeepTreeEchoBot configuration (CPU optimized for better performance)
    bot_config = BotConfig.get_cpu_optimized_config()
    bot_config.enable_debug_mode = True  # Enable debug mode for development
    
    # Create Delta-Chat configuration
    # In production, you would load this from a config file or environment variables
    deltachat_config = DeltaChatConfig(
        email=os.getenv("DELTACHAT_EMAIL", "your-bot@example.com"),
        password=os.getenv("DELTACHAT_PASSWORD", "your-password"),
        bot_name="DeepTreeEchoBot",
        command_prefix="/",
        auto_accept_chats=True,
        respond_to_groups=True,
        respond_to_private=True,
        max_message_length=2000,
        response_timeout=30,
        enabled_commands=["help", "process", "search", "info", "status", "ping"],
        admin_commands=["stats"],
        admin_contacts=os.getenv("DELTACHAT_ADMIN_CONTACTS", "").split(",") if os.getenv("DELTACHAT_ADMIN_CONTACTS") else [],
        db_path="deltachat_bot.db",
        blob_dir="deltachat_blobs"
    )
    
    # Validate configuration
    try:
        deltachat_config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set DELTACHAT_EMAIL and DELTACHAT_PASSWORD environment variables")
        return
    
    # Create and run the bot
    deltachat_bot = DeltaChatBot(bot_config, deltachat_config)
    
    try:
        print("🚀 Starting Delta-Chat bot...")
        print(f"📧 Email: {deltachat_config.email}")
        print(f"🤖 Bot Name: {deltachat_config.bot_name}")
        print("Press Ctrl+C to stop")
        
        # Initialize and run the bot
        await deltachat_bot.initialize()
        await deltachat_bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error running Delta-Chat bot: {e}")
        if bot_config.enable_debug_mode:
            import traceback
            traceback.print_exc()
    finally:
        await deltachat_bot.shutdown()
        print("✅ Bot shutdown complete")


if __name__ == "__main__":
    # Check if deltachat is available
    try:
        import deltachat
    except ImportError:
        print("❌ Delta-Chat integration not available.")
        print("Install with: pip install deltachat")
        exit(1)
    
    # Run the bot
    asyncio.run(main())