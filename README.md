# DebiasML

A practical, explainable and effective approach to reducing bias in machine learning algorithms.

## Overview

### Problem Statement

Machine learning is being used at an ever increasing rate to automate decisions that were previously made by humans. Decisions on job applications, college admissions  and sentencing guidelines (to name a few) can have life changing consequences for people so it’s important to be fair but the argument for reducing bias is not just an ethical one, it’s a financial one too. Multiple studies have found that teams that are more diverse, significantly outperform teams that aren’t. In reducing bias in our algorithms we seek not just to replicate past performance but exceed it.

### Solution

In DebiasML, I have developed a practical and explainable solution through novel application of oversampling. Though popular for data imbalance problems, oversampling has not been adopted to address bias. When tested on the Adult UCI dataset, DebiasML outperforms the state of the art (GANs) on many dimensions. It results in a significantly higher F1 score (as much as +17%) whilst being equally accurate; training and inference are significantly cheaper; it is model agnostic, transparent and by construction improves diversity in its predictions.

The graphic below shows the distribution of predictions on the test set as it changes with the oversampling factor along with performance and bias metrics:
![oversample gif](https://github.com/leenamurgai/debias-ml/blob/master/static/oversample.gif)

### Resource list

- **Blog post** explaining the problem, solution approach and results will be linked to here when available
- [**Presentation slides**](http://bit.ly/debias-ml-slides) explaining the problem, solution approach and results in 5 mins are available here
- [**Presentation recording**](http://bit.ly/debias-ml-recording) explaining
- [**Streamlit report**](http://share.streamlit.io/0.25.0-cdyb/index.html?id=HpMQLQaCFmL4p2dgA86Wz9) showing data exploration and results can be found here

## Running the code on your machine

### Requisites

This repo uses conda's virtual environment for Python 3

Install (mini)conda if not yet installed:

For MacOS:
```shell
$ wget http://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh
$ chmod +x miniconda.sh
$ ./miniconda.sh -b
```

cd into the directory and create the conda virtual environment from environment.yml
```shell
$ conda env create -f environment.yml
```

Activate the virtual environment:
```shell
$ source activate debias-ml
```

- anaconda
- Python 3.6 (Keras/tensorflow does not currently work with Python 3.7)
- [Streamlit](streamlit.io)

### Running the code

cd into the source directory and call
```shell
$ python adult_uci.py
```
