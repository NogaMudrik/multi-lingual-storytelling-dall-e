# -*- coding: utf-8 -*-
"""
@author: noga mudrik
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    author="Noga Mudrik",

    name="multi-lingual-storytelling-dall-e",
    version="0.0.01",
    
    author_email="<nmudrik1@jhmi.edu>",
    description="The multi-lingual DALL-E for story visualization model as described in 'Mudrik, N., Charles, A., “Multi-Lingual DALL-E Storytime”. Arxiv. (2022). https://arxiv.org/abs/2212.11985'",
    long_description=long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.8",
    install_requires = ['numpy', 'matplotlib','google','openai', 'PIL','requests','keras_ocr', 'opencv', ]
)

