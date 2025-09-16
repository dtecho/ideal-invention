"""
Delta-Chat configuration for DeepTreeEchoBot integration.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os


@dataclass
class DeltaChatConfig:
    """Configuration for Delta-Chat integration."""
    
    # Email account settings
    email: Optional[str] = None
    password: Optional[str] = None
    
    # IMAP/SMTP settings (optional - Delta-Chat can auto-configure)
    imap_server: Optional[str] = None
    imap_port: Optional[int] = None
    imap_security: Optional[str] = None  # "ssl", "starttls", or "plain"
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_security: Optional[str] = None
    
    # Bot behavior settings
    bot_name: str = "DeepTreeEchoBot"
    command_prefix: str = "/"
    auto_accept_chats: bool = True
    respond_to_groups: bool = True
    respond_to_private: bool = True
    
    # Processing settings
    max_message_length: int = 2000
    response_timeout: int = 30
    enable_typing_indicator: bool = True
    
    # Command settings
    enabled_commands: List[str] = field(default_factory=lambda: [
        "help", "process", "search", "info", "status"
    ])
    admin_commands: List[str] = field(default_factory=lambda: [
        "shutdown", "restart", "stats", "config"
    ])
    admin_contacts: List[str] = field(default_factory=list)
    
    # Database and storage
    db_path: str = "deltachat_bot.db"
    blob_dir: str = "deltachat_blobs"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeltaChatConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'email': self.email,
            'password': self.password,
            'imap_server': self.imap_server,
            'imap_port': self.imap_port,
            'imap_security': self.imap_security,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'smtp_security': self.smtp_security,
            'bot_name': self.bot_name,
            'command_prefix': self.command_prefix,
            'auto_accept_chats': self.auto_accept_chats,
            'respond_to_groups': self.respond_to_groups,
            'respond_to_private': self.respond_to_private,
            'max_message_length': self.max_message_length,
            'response_timeout': self.response_timeout,
            'enable_typing_indicator': self.enable_typing_indicator,
            'enabled_commands': self.enabled_commands,
            'admin_commands': self.admin_commands,
            'admin_contacts': self.admin_contacts,
            'db_path': self.db_path,
            'blob_dir': self.blob_dir,
        }
        
    @classmethod
    def from_env(cls) -> 'DeltaChatConfig':
        """Create config from environment variables."""
        return cls(
            email=os.getenv('DELTACHAT_EMAIL'),
            password=os.getenv('DELTACHAT_PASSWORD'),
            imap_server=os.getenv('DELTACHAT_IMAP_SERVER'),
            imap_port=int(os.getenv('DELTACHAT_IMAP_PORT', 0)) or None,
            imap_security=os.getenv('DELTACHAT_IMAP_SECURITY'),
            smtp_server=os.getenv('DELTACHAT_SMTP_SERVER'),
            smtp_port=int(os.getenv('DELTACHAT_SMTP_PORT', 0)) or None,
            smtp_security=os.getenv('DELTACHAT_SMTP_SECURITY'),
            bot_name=os.getenv('DELTACHAT_BOT_NAME', 'DeepTreeEchoBot'),
            command_prefix=os.getenv('DELTACHAT_COMMAND_PREFIX', '/'),
            auto_accept_chats=os.getenv('DELTACHAT_AUTO_ACCEPT', 'true').lower() == 'true',
            respond_to_groups=os.getenv('DELTACHAT_RESPOND_GROUPS', 'true').lower() == 'true',
            respond_to_private=os.getenv('DELTACHAT_RESPOND_PRIVATE', 'true').lower() == 'true',
            max_message_length=int(os.getenv('DELTACHAT_MAX_MESSAGE_LENGTH', 2000)),
            response_timeout=int(os.getenv('DELTACHAT_RESPONSE_TIMEOUT', 30)),
            enable_typing_indicator=os.getenv('DELTACHAT_TYPING_INDICATOR', 'true').lower() == 'true',
            admin_contacts=os.getenv('DELTACHAT_ADMIN_CONTACTS', '').split(',') if os.getenv('DELTACHAT_ADMIN_CONTACTS') else [],
            db_path=os.getenv('DELTACHAT_DB_PATH', 'deltachat_bot.db'),
            blob_dir=os.getenv('DELTACHAT_BLOB_DIR', 'deltachat_blobs'),
        )
        
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.email:
            raise ValueError("Email is required for Delta-Chat configuration")
        if not self.password:
            raise ValueError("Password is required for Delta-Chat configuration")
        return True