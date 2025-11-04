#!/usr/bin/env python3
"""
Setup script for LLM Evaluation Framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="llm-evaluation-framework",
    version="1.0.0",
    author="LLM Evaluation Team",
    description="A comprehensive framework for evaluating LLMs on various benchmarks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'llm-eval=main:cli',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="llm evaluation benchmark mmlu aime ifeval vllm",
    project_urls={
        "Documentation": "https://github.com/your-org/llm-evaluation-framework",
        "Source": "https://github.com/your-org/llm-evaluation-framework",
        "Tracker": "https://github.com/your-org/llm-evaluation-framework/issues",
    },
)