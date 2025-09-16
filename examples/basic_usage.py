#!/usr/bin/env python3
"""
Basic usage example for DeepTreeEchoBot.
Demonstrates how to use the bot for simple task processing.
"""
import asyncio
import logging
from deep_tree_echo_bot import DeepTreeEchoBot, BotConfig


async def basic_example():
    """Run a basic example of the DeepTreeEchoBot."""
    print("DeepTreeEchoBot Basic Usage Example")
    print("=" * 40)
    
    # Create a CPU-optimized configuration
    config = BotConfig.get_cpu_optimized_config()
    config.log_level = "INFO"
    
    # Create the bot
    bot = DeepTreeEchoBot(config)
    
    try:
        # Initialize the bot
        print("Initializing bot...")
        await bot.initialize()
        print("Bot initialized successfully!")
        
        # Define some example tasks
        tasks = [
            "Find information about artificial intelligence",
            "Search for recent developments in renewable energy",
            "Explain how machine learning works",
            "Compare different programming languages",
        ]
        
        # Process each task
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Task {i}: {task} ---")
            
            result = await bot.process_task(task)
            
            print(f"Success: {result['success']}")
            print(f"Total Reward: {result['total_reward']:.4f}")
            print(f"Steps Taken: {result['steps_taken']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            
            if result.get('result'):
                print(f"Result Preview: {result['result'][:200]}...")
                
            # Show some episode statistics
            episode_data = result.get('episode_data', {})
            if episode_data.get('steps'):
                step_types = [step.get('info', {}).get('action_type', 'unknown') 
                            for step in episode_data['steps']]
                print(f"Action sequence: {' -> '.join(step_types)}")
        
        # Show final statistics
        print("\n" + "=" * 40)
        print("SESSION SUMMARY")
        print("=" * 40)
        
        rl_stats = bot.rl_agent.get_training_stats()
        print(f"Training steps completed: {rl_stats['training_step']}")
        print(f"Current exploration rate: {rl_stats['epsilon']:.4f}")
        print(f"Experience buffer size: {rl_stats['buffer_size']}")
        
        reasoning_stats = bot.task_processor.get_reasoning_stats()
        print(f"Active tasks: {reasoning_stats['active_tasks']}")
        print(f"Completed tasks: {reasoning_stats['completed_tasks']}")
        
        vector_stats = bot.task_vector_gen.get_vector_stats()
        print(f"Tasks processed: {vector_stats['total_tasks_processed']}")
        print(f"Learned patterns: {vector_stats['learned_patterns']}")
        
        reward_stats = bot.reward_calculator.get_reward_history_stats()
        print(f"Average reward: {reward_stats.get('recent_average', 0):.4f}")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("\nShutting down bot...")
        await bot.shutdown()
        print("Example completed!")


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the example
    asyncio.run(basic_example())