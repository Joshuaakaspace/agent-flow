from setuptools import setup, find_packages

setup(
    name="agent-flow-scheduler",
    version="0.1.0",
    description="Workflow-aware scheduling for multi-agent LLM pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="agent-flow-scheduler",
    url="https://github.com/joeajiteshvarun/agent-flow-scheduler",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    extras_require={
        "openai":    ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.25.0"],
        "dev":       ["pytest>=7.0.0", "pytest-timeout>=2.0.0"],
        "viz":       ["matplotlib>=3.7.0", "networkx>=3.0"],
        "all":       ["openai>=1.0.0", "anthropic>=0.25.0",
                      "matplotlib>=3.7.0", "networkx>=3.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="llm agents scheduling inference agentic workflow",
)
