"""
Message handler for Delta-Chat integration.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

try:
    import deltachat
except ImportError:
    deltachat = None

from .config import DeltaChatConfig
from .commands import CommandProcessor, CommandContext
from ..core.bot import DeepTreeEchoBot


class MessageHandler:
    """Handles incoming messages from Delta-Chat."""
    
    def __init__(self, deep_tree_bot: DeepTreeEchoBot, deltachat_config: DeltaChatConfig):
        self.deep_tree_bot = deep_tree_bot
        self.config = deltachat_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize command processor
        self.command_processor = CommandProcessor(deep_tree_bot, deltachat_config.command_prefix)
        
        # Track active chats and processing status
        self.active_chats: Set[int] = set()
        self.processing_messages: Set[str] = set()  # Track message IDs being processed
        
        # Rate limiting
        self.last_message_time: Dict[str, datetime] = {}
        self.message_count: Dict[str, int] = {}
        
    async def handle_message(self, account, message) -> None:
        """Handle an incoming Delta-Chat message."""
        try:
            # Skip if deltachat is not available
            if not deltachat:
                self.logger.error("deltachat package not available")
                return
                
            # Get message details
            chat = message.chat
            sender = message.get_sender_contact()
            text = message.text
            
            # Skip empty messages or system messages
            if not text or message.is_system_message():
                return
                
            # Skip our own messages
            if sender.addr == account.get_config("addr"):
                return
                
            message_id = f"{chat.id}_{message.id}"
            
            # Skip if already processing this message
            if message_id in self.processing_messages:
                return
                
            # Add to processing set
            self.processing_messages.add(message_id)
            
            try:
                # Check rate limits
                if not self._check_rate_limit(sender.addr):
                    self.logger.warning(f"Rate limit exceeded for {sender.addr}")
                    return
                    
                # Auto-accept chat if configured
                if self.config.auto_accept_chats and chat.is_contact_request():
                    chat.accept()
                    
                # Check if we should respond to this chat type
                if not self._should_respond(chat):
                    return
                    
                # Add chat to active chats
                self.active_chats.add(chat.id)
                
                # Show typing indicator if enabled
                if self.config.enable_typing_indicator:
                    # Note: Delta-Chat doesn't have a direct typing indicator API
                    # This is a placeholder for future implementation
                    pass
                    
                # Check if user is admin
                is_admin = sender.addr in self.config.admin_contacts
                
                # Process the message
                response = await self._process_message(text, sender.addr, chat, is_admin)
                
                # Send response if we have one
                if response:
                    # Truncate response if too long
                    if len(response) > self.config.max_message_length:
                        response = response[:self.config.max_message_length - 20] + "\n\n[Message truncated]"
                        
                    # Send the response
                    chat.send_text(response)
                    self.logger.info(f"Sent response to {sender.addr} in chat {chat.id}")
                    
            finally:
                # Remove from processing set
                self.processing_messages.discard(message_id)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            # Try to send error message if possible
            try:
                if 'chat' in locals() and chat:
                    chat.send_text("❌ Sorry, I encountered an error processing your message.")
            except:
                pass
                
    def _check_rate_limit(self, sender_addr: str) -> bool:
        """Check if sender is within rate limits."""
        now = datetime.now()
        
        # Simple rate limiting: max 10 messages per minute per user
        if sender_addr not in self.last_message_time:
            self.last_message_time[sender_addr] = now
            self.message_count[sender_addr] = 1
            return True
            
        time_diff = (now - self.last_message_time[sender_addr]).total_seconds()
        
        # Reset counter if more than a minute has passed
        if time_diff > 60:
            self.last_message_time[sender_addr] = now
            self.message_count[sender_addr] = 1
            return True
            
        # Check current count
        current_count = self.message_count.get(sender_addr, 0)
        if current_count >= 10:
            return False
            
        # Increment count
        self.message_count[sender_addr] = current_count + 1
        return True
        
    def _should_respond(self, chat) -> bool:
        """Check if we should respond to this chat."""
        if not deltachat:
            return False
            
        # Check group vs private chat settings
        if chat.is_group():
            return self.config.respond_to_groups
        else:
            return self.config.respond_to_private
            
    async def _process_message(self, text: str, sender_addr: str, chat, is_admin: bool) -> Optional[str]:
        """Process a message and return response."""
        text = text.strip()
        
        # Check if it's a command
        if self.command_processor.is_command(text):
            command, args = self.command_processor.parse_command(text)
            
            # Create command context
            context = CommandContext(
                command=command,
                args=args,
                message_text=text,
                sender_email=sender_addr,
                chat_id=chat.id,
                is_group=chat.is_group(),
                is_admin=is_admin
            )
            
            # Process command
            return await self.command_processor.process_command(context)
            
        else:
            # Not a command - treat as natural language interaction
            return await self._process_natural_language(text, sender_addr, chat)
            
    async def _process_natural_language(self, text: str, sender_addr: str, chat) -> Optional[str]:
        """Process natural language message."""
        # For now, provide a helpful response
        greeting_keywords = ['hello', 'hi', 'hey', 'greetings']
        question_keywords = ['what', 'how', 'why', 'when', 'where', 'who']
        
        text_lower = text.lower()
        
        # Check for greetings
        if any(word in text_lower for word in greeting_keywords):
            return (
                f"Hello! 👋 I'm DeepTreeEchoBot, an AI assistant.\n\n"
                f"I can help you with various tasks. Type {self.config.command_prefix}help to see available commands, "
                f"or use {self.config.command_prefix}process followed by your task description."
            )
            
        # Check for questions
        if any(word in text_lower for word in question_keywords):
            return (
                f"I can help answer questions! Use {self.config.command_prefix}process followed by your question, "
                f"or {self.config.command_prefix}search to search for information."
            )
            
        # For other messages, suggest using commands
        if len(text) > 10:  # Only for substantial messages
            return (
                f"I'm an AI assistant that works best with commands. "
                f"Try {self.config.command_prefix}process {text[:50]}{'...' if len(text) > 50 else ''} "
                f"or type {self.config.command_prefix}help for more options."
            )
            
        return None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get message handler statistics."""
        return {
            'active_chats': len(self.active_chats),
            'processing_messages': len(self.processing_messages),
            'tracked_users': len(self.last_message_time),
        }