from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="dynoframe",
    version="0.1.0",
    author="izoon",
    author_email="author@example.com",  # Replace with actual email if desired
    description="A framework for building dynamic agent-based systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/izoon/dynoframe",  # Updated to the actual GitHub URL
    packages=find_packages(),
    package_data={
        "dynoframe": ["*.py"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="agents, llm, rag, ai, framework",
) 