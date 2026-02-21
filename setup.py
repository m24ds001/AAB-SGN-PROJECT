from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aab-sgn",
    version="1.0.0",
    author="Ramsha Mehreen, Renikunta Ramesh",
    author_email="ramshamehreen2208@gmail.com",
    description="Hesitation Margins are Necessary:
Ambiguity-Aware Backpropagation for Robust
Learning Under Label Noise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-org]/AAB-SGN",
    project_urls={
        "Bug Tracker": "https://github.com/[your-org]/AAB-SGN/issues",
        "Paper": "link-to-paper",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.10.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
        ],
    },
)
