# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open('README.md', 'r') as f:
  readme =  f.read()

requirements = [
    "torch>=1.5.0",
    "transformers>=3.5.0",
    "sentencepiece==0.1.91",
    "emoji",
    "pythainlp>=2.2.4",
    "tqdm",
    "datasets>=1.2.1",
    "sefr_cut",
    "seqeval",
    "pytorch-lightning",
    "pydantic",
    "jsonlines",
    "tokenizers==0.9.3",
    "pandas",
]


setup(
    name="thai2transformers",
    version="0.1.3",
    description="Pretraining transformer based Thai language models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="VISTEC-depa AI Research Institute of Thailand",
    author_email="",
    url="https://github.com/vistec-AI/thai2transformers",
    packages=find_packages(exclude=["notebooks", "notebooks.*", "external_scripts", "external_scripts.*"]),
    python_requires=">=3.6",
    package_data={},
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "thainlp",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "ThaiNLP",
        "Thai NLP",
        "Thai language",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: Thai",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Documentation": "https://github.com/vistec-AI/thai2transformers",
        "Tutorials": "https://colab.research.google.com/drive/1Kbk6sBspZLwcnOE61adAQo30xxqOQ9ko?usp=sharing",
        "Source Code": "https://github.com/vistec-AI/thai2transformers",
        "Bug Tracker": "https://github.com/vistec-AI/thai2transformers/issues",
    },
)
