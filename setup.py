from setuptools import setup, find_packages

setup(
    name="ottominer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'rich>=10.0.0',
        'pyyaml>=6.0.0',
        'psutil>=5.9.0',
        'pymupdf4llm>=0.0.17',
        'chardet>=5.0.0',
        'durak-nlp>=0.1.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.0.0',
        'matplotlib>=3.7.0',
    ],
    extras_require={
        'ocr': ['surya-ocr>=0.4.0'],
        'ollama': ['ollama>=0.1.0'],
        'embeddings': ['sentence-transformers>=2.0.0'],
        'dev': [
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'pytest>=7.0.0',
            'pytest-timeout>=2.1.0',
            'pytest-cov>=4.1.0',
            'reportlab>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ottominer=ottominer.cli.main:main',
        ],
    },
)
