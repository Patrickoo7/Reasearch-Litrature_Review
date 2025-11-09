from setuptools import setup, find_packages

setup(
    name="research-reproducer",
    version="0.1.0",
    description="A tool for reproducing research papers by automatically finding and running associated code",
    author="Research Community",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.10.0",
        "PyGithub>=2.1.0",
        "arxiv>=2.0.0",
        "scholarly>=1.7.0",
        "gitpython>=3.1.0",
        "docker>=6.1.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "inquirer>=3.1.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "research-reproduce=research_reproducer.cli:main",
        ],
    },
    python_requires=">=3.8",
)
