from setuptools import setup, find_packages

setup(
    name="smart-campus-energy-optimizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered energy optimization system for campuses",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-campus-energy-optimizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "scikit-learn>=1.3.0",
    ],
)
