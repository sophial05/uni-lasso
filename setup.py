from setuptools import setup, find_packages

setup(
    name="unilasso",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "scikit-learn",
        "statsmodels",
        "adelie",
        "matplotlib",
        "colorama",
        "pytest",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black"],
    },
    author="Sophia Lu",
    author_email="sophialu@stanford.sedu",
    description="A package for UniLasso: Univariate-Guided Sparse Regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sophial05/uni-lasso",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)