from setuptools import setup, find_packages

setup(
    name="deep-tree-echo-bot",
    version="0.1.0",
    description="Enhanced deep-tree-echo-bot with reinforcement learning",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "transformers>=4.20.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "aiohttp>=3.8.0",
        "click>=8.1.0",
        "pydantic>=1.10.0",
        "typing-extensions>=4.3.0",
        "deltachat>=1.140.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "deep-tree-echo-bot=deep_tree_echo_bot.cli:main",
        ],
    },
)