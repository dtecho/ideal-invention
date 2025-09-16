# DeepTreeEchoBot - Enhanced AI Agent

An advanced deep-tree-echo-bot implementation using machine-generated task vectors, smooth reward functions, and pure reinforcement learning for optimized thinking processes.

## Features

- **Strong Agentic Search**: Intelligent multi-engine search with semantic ranking
- **Basic Browsing Capabilities**: CPU-optimized web content retrieval and processing
- **CPU Performance**: Designed to run efficiently on CPU hardware with decent speed
- **Focused Reasoning**: Systematic task decomposition and step-by-step processing
- **Machine-Generated Task Vectors**: Adaptive task representations that optimize for specific requirements
- **Smooth Reward Functions**: Multi-category reward system across task completion, efficiency, quality, exploration, learning, and coherence
- **Pure Reinforcement Learning**: No supervised fine-tuning, learns purely from environment interactions
- **Delta-Chat Integration**: Native support for Delta-Chat messaging platform (email-based chat)

## Architecture

The bot consists of several key components:

1. **Core Bot** (`DeepTreeEchoBot`): Main orchestrator that coordinates all components
2. **RL Agent**: Deep Q-Network implementation for action selection and learning
3. **Agentic Search Engine**: Multi-engine search with semantic ranking
4. **Task Processor**: Focused reasoning system with task decomposition
5. **Web Browser**: CPU-optimized content retrieval and extraction
6. **Task Vector Generator**: Machine learning-based task representation
7. **Smooth Reward Calculator**: Multi-category reward computation

## Installation

```bash
# Clone the repository
git clone https://github.com/dtecho/ideal-invention.git
cd ideal-invention

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional: Install Delta-Chat integration
pip install deltachat
```

## Quick Start

### Command Line Interface

```bash
# Process a single task
deep-tree-echo-bot process "Find information about renewable energy"

# Train the bot
deep-tree-echo-bot train --num-episodes 20 --save-model models/trained_bot

# Interactive mode
deep-tree-echo-bot interactive

# Generate configuration file
deep-tree-echo-bot generate-config --preset cpu --output config.json

# Run as Delta-Chat bot (requires deltachat package)
deep-tree-echo-bot deltachat --email your-bot@example.com --password your-password

# Generate Delta-Chat configuration
deep-tree-echo-bot generate-deltachat-config --email your-bot@example.com --output deltachat_config.json
```

### Python API

```python
import asyncio
from deep_tree_echo_bot import DeepTreeEchoBot, BotConfig

async def main():
    # Create bot with CPU-optimized configuration
    config = BotConfig.get_cpu_optimized_config()
    bot = DeepTreeEchoBot(config)
    
    # Initialize the bot
    await bot.initialize()
    
    # Process a task
    result = await bot.process_task("Research machine learning trends")
    
    print(f"Success: {result['success']}")
    print(f"Result: {result['result']}")
    print(f"Reward: {result['total_reward']}")
    
    # Shutdown
    await bot.shutdown()

# Run the example
asyncio.run(main())
```

## Delta-Chat Integration

DeepTreeEchoBot now supports integration with Delta-Chat, an email-based chat platform that provides secure, decentralized messaging.

### Quick Start with Delta-Chat

1. **Install Delta-Chat support:**
   ```bash
   pip install deltachat
   ```

2. **Set up bot credentials:**
   ```bash
   export DELTACHAT_EMAIL="your-bot@example.com"
   export DELTACHAT_PASSWORD="your-app-password"  # Use app password for Gmail
   ```

3. **Run the bot:**
   ```bash
   deep-tree-echo-bot deltachat
   ```

### Delta-Chat Bot Commands

Once running, users can interact with the bot using these commands:

- `/help` - Show available commands
- `/process <task>` - Process a task with the AI bot
- `/search <query>` - Search for information  
- `/info` - Show bot information
- `/status` - Show bot status
- `/ping` - Check if bot is responsive

### Configuration

Generate a Delta-Chat configuration file:

```bash
deep-tree-echo-bot generate-deltachat-config --email your-bot@example.com --output deltachat_config.json
```

Then run with the configuration:

```bash
deep-tree-echo-bot deltachat --deltachat-config deltachat_config.json
```

### Examples

See the `examples/deltachat/` directory for:
- Configuration templates
- Python integration examples
- Setup instructions
- Troubleshooting guide

## Configuration

The bot supports various configuration options for different use cases:

- **CPU Optimized**: `BotConfig.get_cpu_optimized_config()` - Optimized for CPU performance
- **Development**: `BotConfig.get_development_config()` - Enhanced debugging and faster training
- **Custom**: Create custom configurations by modifying `BotConfig` parameters

## Training

The bot uses pure reinforcement learning without any supervised fine-tuning:

```bash
# Train with default tasks
deep-tree-echo-bot train --num-episodes 50

# Train with custom tasks
deep-tree-echo-bot train --tasks-file my_tasks.txt --num-episodes 100 --save-model trained_model
```

## Reward System

The bot uses a sophisticated multi-category reward system:

- **Task Completion** (25%): Progress and success metrics
- **Efficiency** (20%): Speed and resource utilization
- **Quality** (20%): Output quality and relevance
- **Exploration** (15%): Action diversity and novelty
- **Learning** (10%): Information gain and adaptation
- **Coherence** (10%): Action sequence logic

## Performance

- **CPU Optimized**: Runs efficiently on standard CPU hardware
- **Memory Efficient**: Configurable memory limits and optimization
- **Scalable**: Adjustable complexity based on available resources

## Examples

See the `examples/` directory for:
- Basic usage examples
- Training scripts
- Configuration templates
- Integration examples

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/

# Enable debug mode
deep-tree-echo-bot --debug interactive
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.