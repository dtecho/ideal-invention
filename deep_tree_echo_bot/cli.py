"""
Command Line Interface for DeepTreeEchoBot.
Provides easy access to bot functionality via CLI commands.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .core.bot import DeepTreeEchoBot
from .config.settings import BotConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('deep_tree_echo_bot.log')
        ]
    )


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, log_level, debug):
    """DeepTreeEchoBot - Enhanced AI agent with reinforcement learning."""
    ctx.ensure_object(dict)
    
    # Setup logging
    if debug:
        log_level = 'DEBUG'
    setup_logging(log_level)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_dict = json.load(f)
        bot_config = BotConfig.from_dict(config_dict)
    else:
        bot_config = BotConfig.get_cpu_optimized_config()
        
    if debug:
        bot_config.enable_debug_mode = True
        bot_config.log_level = 'DEBUG'
        
    ctx.obj['config'] = bot_config


@cli.command()
@click.argument('task')
@click.option('--context', '-ctx', help='Task context as JSON string')
@click.option('--save-result', '-s', type=click.Path(), help='Save result to file')
@click.pass_context
def process(ctx, task, context, save_result):
    """Process a single task using the DeepTreeEchoBot."""
    config = ctx.obj['config']
    
    async def run_task():
        bot = DeepTreeEchoBot(config)
        
        try:
            await bot.initialize()
            
            # Parse context if provided
            task_context = None
            if context:
                try:
                    task_context = json.loads(context)
                except json.JSONDecodeError:
                    click.echo(f"Warning: Invalid JSON context, using as string: {context}")
                    task_context = {'context': context}
                    
            # Process the task
            click.echo(f"Processing task: {task}")
            result = await bot.process_task(task, task_context)
            
            # Display results
            click.echo("\n" + "="*50)
            click.echo("TASK PROCESSING RESULTS")
            click.echo("="*50)
            click.echo(f"Success: {result['success']}")
            click.echo(f"Total Reward: {result['total_reward']:.4f}")
            click.echo(f"Steps Taken: {result['steps_taken']}")
            click.echo(f"Duration: {result['duration']:.2f} seconds")
            
            if result.get('result'):
                click.echo(f"\nResult:\n{result['result']}")
                
            # Save result if requested
            if save_result:
                with open(save_result, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                click.echo(f"\nResult saved to {save_result}")
                
        except KeyboardInterrupt:
            click.echo("\nTask interrupted by user")
        except Exception as e:
            click.echo(f"Error processing task: {e}")
            if config.enable_debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            await bot.shutdown()
            
    asyncio.run(run_task())


@cli.command()
@click.option('--tasks-file', '-f', type=click.Path(exists=True), help='File containing tasks (one per line)')
@click.option('--num-episodes', '-n', default=10, help='Number of training episodes')
@click.option('--save-model', '-m', type=click.Path(), help='Save trained model to path')
@click.pass_context
def train(ctx, tasks_file, num_episodes, save_model):
    """Train the bot using multiple tasks."""
    config = ctx.obj['config']
    
    async def run_training():
        bot = DeepTreeEchoBot(config)
        
        try:
            await bot.initialize()
            
            # Load tasks
            if tasks_file:
                with open(tasks_file, 'r') as f:
                    tasks = [line.strip() for line in f if line.strip()]
            else:
                # Default training tasks
                tasks = [
                    "Find information about artificial intelligence",
                    "Search for recent developments in machine learning",
                    "Analyze the benefits of renewable energy",
                    "Compare different programming languages",
                    "Explain how neural networks work",
                    "Research the history of computing",
                    "Find solutions for climate change",
                    "Analyze market trends in technology",
                    "Search for best practices in software development",
                    "Explain quantum computing concepts"
                ]
                
            click.echo(f"Starting training with {len(tasks)} tasks for {num_episodes} episodes")
            
            total_rewards = []
            
            for episode in range(num_episodes):
                episode_rewards = []
                
                click.echo(f"\nEpisode {episode + 1}/{num_episodes}")
                click.echo("-" * 30)
                
                for i, task in enumerate(tasks):
                    click.echo(f"Task {i + 1}/{len(tasks)}: {task[:50]}...")
                    
                    result = await bot.process_task(task)
                    episode_rewards.append(result['total_reward'])
                    
                    click.echo(f"  Reward: {result['total_reward']:.4f}, Steps: {result['steps_taken']}")
                    
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                total_rewards.append(avg_reward)
                
                click.echo(f"Episode {episode + 1} average reward: {avg_reward:.4f}")
                
                # Save checkpoint
                if save_model and (episode + 1) % config.checkpoint_frequency == 0:
                    checkpoint_path = f"{save_model}_ep{episode + 1}.pt"
                    bot.rl_agent.save_model(checkpoint_path)
                    click.echo(f"Checkpoint saved: {checkpoint_path}")
                    
            # Training summary
            click.echo("\n" + "="*50)
            click.echo("TRAINING SUMMARY")
            click.echo("="*50)
            click.echo(f"Episodes completed: {num_episodes}")
            click.echo(f"Average reward: {sum(total_rewards) / len(total_rewards):.4f}")
            click.echo(f"Best episode reward: {max(total_rewards):.4f}")
            click.echo(f"Final episode reward: {total_rewards[-1]:.4f}")
            
            # Save final model
            if save_model:
                final_path = f"{save_model}_final.pt"
                bot.rl_agent.save_model(final_path)
                click.echo(f"Final model saved: {final_path}")
                
        except KeyboardInterrupt:
            click.echo("\nTraining interrupted by user")
        except Exception as e:
            click.echo(f"Error during training: {e}")
            if config.enable_debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            await bot.shutdown()
            
    asyncio.run(run_training())


@cli.command()
@click.pass_context
def interactive(ctx):
    """Run the bot in interactive mode."""
    config = ctx.obj['config']
    
    async def run_interactive():
        bot = DeepTreeEchoBot(config)
        
        try:
            await bot.initialize()
            
            click.echo("DeepTreeEchoBot Interactive Mode")
            click.echo("Type 'quit' or 'exit' to stop")
            click.echo("Type 'help' for available commands")
            click.echo("-" * 40)
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    
                    if user_input.lower() == 'help':
                        click.echo("\nAvailable commands:")
                        click.echo("  help - Show this help message")
                        click.echo("  stats - Show bot statistics")
                        click.echo("  clear - Clear conversation history")
                        click.echo("  quit/exit - Exit interactive mode")
                        click.echo("\nOr enter any task for the bot to process")
                        continue
                        
                    if user_input.lower() == 'stats':
                        # Show statistics
                        rl_stats = bot.rl_agent.get_training_stats()
                        click.echo(f"\nBot Statistics:")
                        click.echo(f"  Training steps: {rl_stats['training_step']}")
                        click.echo(f"  Episodes: {rl_stats['episode_count']}")
                        click.echo(f"  Epsilon: {rl_stats['epsilon']:.4f}")
                        click.echo(f"  Buffer size: {rl_stats['buffer_size']}")
                        continue
                        
                    if user_input.lower() == 'clear':
                        # Clear caches and history
                        await bot.web_browser.clear_cache()
                        click.echo("History cleared")
                        continue
                        
                    if not user_input:
                        continue
                        
                    # Process the task
                    click.echo(f"Processing: {user_input}")
                    result = await bot.process_task(user_input)
                    
                    # Display result
                    click.echo(f"\nSuccess: {result['success']}")
                    click.echo(f"Reward: {result['total_reward']:.4f}")
                    click.echo(f"Steps: {result['steps_taken']}")
                    
                    if result.get('result'):
                        click.echo(f"\nResult:\n{result['result']}")
                        
                except KeyboardInterrupt:
                    click.echo("\nUse 'quit' or 'exit' to stop")
                except EOFError:
                    break
                    
        except Exception as e:
            click.echo(f"Error in interactive mode: {e}")
            if config.enable_debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            await bot.shutdown()
            
    asyncio.run(run_interactive())


@cli.command()
@click.option('--deltachat-config', '-dc', type=click.Path(exists=True), help='Delta-Chat configuration file')
@click.option('--email', '-e', help='Delta-Chat email address')
@click.option('--password', '-p', help='Delta-Chat password')
@click.option('--db-path', help='Delta-Chat database path')
@click.pass_context
def deltachat(ctx, deltachat_config, email, password, db_path):
    """Run the bot in Delta-Chat mode."""
    try:
        from .deltachat_integration import DeltaChatBot, DeltaChatConfig
    except ImportError:
        click.echo("❌ Delta-Chat integration not available. Install with: pip install deltachat")
        return
        
    config = ctx.obj['config']
    
    # Load Delta-Chat configuration
    if deltachat_config:
        with open(deltachat_config, 'r') as f:
            dc_config_dict = json.load(f)
        dc_config = DeltaChatConfig.from_dict(dc_config_dict)
    else:
        # Create config from command line args or environment
        dc_config = DeltaChatConfig.from_env()
        
        # Override with command line arguments
        if email:
            dc_config.email = email
        if password:
            dc_config.password = password
        if db_path:
            dc_config.db_path = db_path
            
    # Validate configuration
    try:
        dc_config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        click.echo("Use --email and --password, or set DELTACHAT_EMAIL and DELTACHAT_PASSWORD environment variables")
        return
        
    async def run_deltachat_bot():
        deltachat_bot = DeltaChatBot(config, dc_config)
        
        try:
            click.echo("🚀 Starting Delta-Chat bot...")
            click.echo(f"📧 Email: {dc_config.email}")
            click.echo(f"🤖 Bot Name: {dc_config.bot_name}")
            click.echo("Press Ctrl+C to stop")
            
            await deltachat_bot.initialize()
            await deltachat_bot.run()
            
        except KeyboardInterrupt:
            click.echo("\n🛑 Shutdown requested by user")
        except Exception as e:
            click.echo(f"❌ Error running Delta-Chat bot: {e}")
            if config.enable_debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            await deltachat_bot.shutdown()
            
    asyncio.run(run_deltachat_bot())


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for configuration')
@click.option('--preset', type=click.Choice(['cpu', 'development', 'default']), default='cpu', help='Configuration preset')
def generate_config(output, preset):
    """Generate a configuration file."""
    if preset == 'cpu':
        config = BotConfig.get_cpu_optimized_config()
    elif preset == 'development':
        config = BotConfig.get_development_config()
    else:
        config = BotConfig()
        
    config_dict = config.to_dict()
    
    if output:
        with open(output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        click.echo(f"Configuration saved to {output}")
    else:
        click.echo(json.dumps(config_dict, indent=2))


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for configuration')
@click.option('--email', '-e', help='Delta-Chat email address')
@click.option('--preset', type=click.Choice(['basic', 'advanced']), default='basic', help='Configuration preset')
def generate_deltachat_config(output, email, preset):
    """Generate a Delta-Chat configuration file."""
    try:
        from .deltachat_integration import DeltaChatConfig
    except ImportError:
        click.echo("❌ Delta-Chat integration not available. Install with: pip install deltachat")
        return
        
    if preset == 'basic':
        config = DeltaChatConfig(
            email=email or "your-bot@example.com",
            password="your-password",
            bot_name="DeepTreeEchoBot",
            auto_accept_chats=True,
            enabled_commands=["help", "process", "search", "info", "status", "ping"]
        )
    else:  # advanced
        config = DeltaChatConfig(
            email=email or "your-bot@example.com", 
            password="your-password",
            bot_name="DeepTreeEchoBot",
            auto_accept_chats=True,
            respond_to_groups=True,
            respond_to_private=True,
            max_message_length=2000,
            response_timeout=30,
            enabled_commands=["help", "process", "search", "info", "status", "ping"],
            admin_commands=["stats", "shutdown"],
            admin_contacts=["admin@example.com"]
        )
        
    config_dict = config.to_dict()
    
    if output:
        with open(output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        click.echo(f"Delta-Chat configuration saved to {output}")
    else:
        click.echo(json.dumps(config_dict, indent=2))


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()