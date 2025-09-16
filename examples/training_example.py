#!/usr/bin/env python3
"""
Training example for DeepTreeEchoBot.
Demonstrates the reinforcement learning training process.
"""
import asyncio
import logging
from pathlib import Path
from deep_tree_echo_bot import DeepTreeEchoBot, BotConfig

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


async def training_example():
    """Run a training example with the DeepTreeEchoBot."""
    print("DeepTreeEchoBot Training Example")
    print("=" * 40)
    
    # Create a development configuration for faster training
    config = BotConfig.get_development_config()
    config.log_level = "INFO"
    
    # Create the bot
    bot = DeepTreeEchoBot(config)
    
    try:
        # Initialize the bot
        print("Initializing bot for training...")
        await bot.initialize()
        print("Bot initialized successfully!")
        
        # Load training tasks
        tasks_file = Path("training_tasks.txt")
        if tasks_file.exists():
            with open(tasks_file, 'r') as f:
                tasks = [line.strip() for line in f if line.strip()]
        else:
            # Fallback tasks
            tasks = [
                "Find information about machine learning",
                "Search for renewable energy solutions", 
                "Analyze the benefits of automation",
                "Compare different AI frameworks",
                "Explain deep learning concepts",
                "Research climate change impacts",
                "Find cybersecurity best practices",
                "Analyze market trends in technology"
            ]
        
        print(f"Training with {len(tasks)} tasks")
        
        # Training parameters
        num_episodes = 5  # Reduced for example
        episode_rewards = []
        episode_steps = []
        
        # Training loop
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            episode_total_reward = 0
            episode_total_steps = 0
            
            # Process each task in the episode
            for i, task in enumerate(tasks):
                print(f"Task {i + 1}/{len(tasks)}: {task[:40]}...")
                
                result = await bot.process_task(task)
                
                episode_total_reward += result['total_reward']
                episode_total_steps += result['steps_taken']
                
                print(f"  Reward: {result['total_reward']:.4f}, "
                      f"Steps: {result['steps_taken']}, "
                      f"Success: {result['success']}")
                
            # Calculate episode averages
            avg_reward = episode_total_reward / len(tasks)
            avg_steps = episode_total_steps / len(tasks)
            
            episode_rewards.append(avg_reward)
            episode_steps.append(avg_steps)
            
            print(f"Episode {episode + 1} Summary:")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Average Steps: {avg_steps:.2f}")
            
            # Show RL agent progress
            rl_stats = bot.rl_agent.get_training_stats()
            print(f"  Training Steps: {rl_stats['training_step']}")
            print(f"  Exploration Rate: {rl_stats['epsilon']:.4f}")
        
        # Training summary
        print("\n" + "=" * 40)
        print("TRAINING SUMMARY")
        print("=" * 40)
        
        print(f"Episodes completed: {num_episodes}")
        print(f"Final average reward: {episode_rewards[-1]:.4f}")
        print(f"Best episode reward: {max(episode_rewards):.4f}")
        print(f"Reward improvement: {episode_rewards[-1] - episode_rewards[0]:.4f}")
        
        # Display learning progress
        print("\nLearning Progress:")
        for i, (reward, steps) in enumerate(zip(episode_rewards, episode_steps)):
            print(f"  Episode {i+1}: Reward={reward:.4f}, Steps={steps:.1f}")
            
        # Show final bot statistics
        print("\nFinal Bot Statistics:")
        rl_stats = bot.rl_agent.get_training_stats()
        print(f"  Total Training Steps: {rl_stats['training_step']}")
        print(f"  Experience Buffer Size: {rl_stats['buffer_size']}")
        print(f"  Final Exploration Rate: {rl_stats['epsilon']:.4f}")
        
        vector_stats = bot.task_vector_gen.get_vector_stats()
        print(f"  Task Patterns Learned: {vector_stats['learned_patterns']}")
        print(f"  Tasks Processed: {vector_stats['total_tasks_processed']}")
        
        reward_stats = bot.reward_calculator.get_reward_history_stats()
        print(f"  Recent Average Reward: {reward_stats.get('recent_average', 0):.4f}")
        print(f"  Reward Standard Deviation: {reward_stats.get('recent_std', 0):.4f}")
        
        # Try to create a simple plot (if matplotlib is available)
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(range(1, num_episodes + 1), episode_rewards, 'b-o')
                plt.title('Average Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(range(1, num_episodes + 1), episode_steps, 'r-o')
                plt.title('Average Steps per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Average Steps')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
                print(f"\nTraining progress plot saved as 'training_progress.png'")
                
            except Exception as e:
                print(f"\nError creating plot: {e}")
        else:
            print("\nMatplotlib not available, skipping plot generation")
        
        # Demonstrate improved performance
        print("\n" + "=" * 40)
        print("TESTING IMPROVED PERFORMANCE")
        print("=" * 40)
        
        test_task = "Find comprehensive information about sustainable technology"
        print(f"Testing with: {test_task}")
        
        result = await bot.process_task(test_task)
        print(f"Final test result:")
        print(f"  Success: {result['success']}")
        print(f"  Total Reward: {result['total_reward']:.4f}")
        print(f"  Steps Taken: {result['steps_taken']}")
        print(f"  Duration: {result['duration']:.2f} seconds")
        
        if result.get('result'):
            print(f"  Result: {result['result'][:200]}...")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("\nShutting down bot...")
        await bot.shutdown()
        print("Training example completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the training example
    asyncio.run(training_example())