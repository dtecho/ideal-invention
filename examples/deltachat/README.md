# Delta-Chat Integration Examples

This directory contains examples for using DeepTreeEchoBot with Delta-Chat integration.

## Files

- `basic_deltachat_config.json` - Basic configuration template for Delta-Chat bot
- `run_deltachat_bot.py` - Example Python script to run the bot programmatically
- `README.md` - This documentation file

## Prerequisites

1. Install Delta-Chat Python bindings:
   ```bash
   pip install deltachat
   ```

2. Have an email account for the bot (Gmail, Yahoo, ProtonMail, etc.)

## Quick Start

### Method 1: Using CLI

1. Generate a configuration file:
   ```bash
   deep-tree-echo-bot generate-deltachat-config --email your-bot@example.com --output deltachat_config.json
   ```

2. Edit the configuration file with your email credentials:
   ```json
   {
     "email": "your-bot@example.com",
     "password": "your-app-password",
     ...
   }
   ```

3. Run the bot:
   ```bash
   deep-tree-echo-bot deltachat --deltachat-config deltachat_config.json
   ```

### Method 2: Using Environment Variables

1. Set environment variables:
   ```bash
   export DELTACHAT_EMAIL="your-bot@example.com"
   export DELTACHAT_PASSWORD="your-app-password"
   export DELTACHAT_BOT_NAME="DeepTreeEchoBot"
   ```

2. Run the bot:
   ```bash
   deep-tree-echo-bot deltachat
   ```

### Method 3: Using Python Script

1. Copy and modify `run_deltachat_bot.py`
2. Set your email credentials in the script or environment variables
3. Run the script:
   ```bash
   python run_deltachat_bot.py
   ```

## Configuration Options

### Email Account Setup

For Gmail accounts, you'll need to:
1. Enable 2-factor authentication
2. Generate an "App Password" instead of using your regular password
3. Use the app password in the configuration

### Bot Commands

The bot supports the following commands by default:

- `/help` - Show available commands
- `/process <task>` - Process a task with the AI bot
- `/search <query>` - Search for information
- `/info` - Show bot information
- `/status` - Show bot status
- `/ping` - Check if bot is responsive

### Admin Commands

If you set yourself as an admin contact, you also get:

- `/stats` - Show detailed bot statistics

### Configuration Parameters

Key configuration options:

- `email` - Bot's email address
- `password` - Email password (use app password for Gmail)
- `bot_name` - Display name for the bot
- `command_prefix` - Command prefix (default: "/")
- `auto_accept_chats` - Auto-accept new chat requests
- `respond_to_groups` - Respond in group chats
- `respond_to_private` - Respond to private messages
- `max_message_length` - Maximum response length
- `admin_contacts` - List of admin email addresses

## Security Considerations

1. **Never commit credentials to version control**
2. Use environment variables or secure config files
3. For Gmail, use app passwords instead of your main password
4. Consider using a dedicated email account for the bot
5. Regularly rotate passwords
6. Monitor bot activity and logs

## Troubleshooting

### Common Issues

1. **"Configuration error: Email is required"**
   - Set DELTACHAT_EMAIL environment variable or use --email flag

2. **"deltachat package not available"**
   - Install with: `pip install deltachat`

3. **Authentication failures**
   - For Gmail: Use app password, not regular password
   - Enable IMAP access in email settings
   - Check firewall/network restrictions

4. **Bot not responding to messages**
   - Check logs for errors
   - Verify email account is working
   - Test with simple commands like `/ping`

### Debug Mode

Enable debug mode for detailed logging:

```bash
deep-tree-echo-bot --debug deltachat --email your-bot@example.com
```

### Logs

The bot logs to both console and `deep_tree_echo_bot.log` file. Check logs for detailed error messages and debugging information.

## Advanced Usage

### Custom Commands

You can extend the bot by modifying the command processor. See the source code in `deep_tree_echo_bot/deltachat_integration/commands.py` for examples.

### Integration with Other Services

The Delta-Chat bot can be extended to integrate with other services by modifying the message handler and adding new command processors.