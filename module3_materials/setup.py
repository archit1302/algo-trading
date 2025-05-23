from setuptools import setup, find_packages

setup(
    name="module3_materials",
    version="0.1.0",
    description="Materials for Module 3: Data Fetching and Processing with Upstox API for Algorithmic Trading",
    author="Archit Mittal",
    author_email="archit@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "pytz>=2021.1",
        "python-dotenv>=0.19.0",
        "websocket-client>=1.2.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)
