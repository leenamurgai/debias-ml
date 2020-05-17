# debias-ml

A practical, explainable and effective approach to reducing bias in machine learning algorithms.

## 1. Overview

### 1.1. Problem Statement

Machine learning is being used at an ever increasing rate to automate decisions that were previously made by humans. Decisions on job applications, college admissions  and sentencing guidelines (to name a few) can have life changing consequences for people so it’s important to be fair. But the argument for reducing bias is not just an ethical one, it’s a financial one too. Multiple studies have found that teams that are more diverse in race and gender, significantly outperform teams that aren’t. In reducing bias in our algorithms we hope not just to replicate past performance but to exceed it.

### 1.2. Solution

In DebiasML, I have developed a practical and explainable solution through novel application of oversampling. Though popular for data imbalance problems, oversampling has not been adopted to address bias. When tested on the Adult UCI dataset, DebiasML outperforms the state of the art (GANs) on many dimensions. It results in a significantly higher F1 score (as much as +17%) whilst being equally accurate; training is ten times faster; it is model agnostic, transparent and by construction improves diversity in its predictions.

The graphic below shows the distribution of predictions on the test set as it changes with the oversampling factor along with performance and bias metrics:
![oversample gif](https://github.com/leenamurgai/debias-ml/blob/master/static/oversample.gif)

## 2. Resource list

- [**Blog post**](http://bit.ly/debias-ml-blog) on Medium
- [**Presentation recording**](http://bit.ly/debias-ml-video) Lightning talk at PyBay2019
- [**Presentation slides**](http://bit.ly/debias-ml-slides)
- **Streamlit reports:**
  - [**Data Analysis Report**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=JDjgoPh55HrSxbKvpthCj2M) showing only the data analysis
  - [**Model Analysis Report**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=UCo7PvitQe3DqdWrz2ZBon) showing only model amalysis
  - [**Oversampling Analysis Report**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=QdPWBFJza6qoAfB1mivUm2) showing only oversampling analysis## 4. Running the code on your machine
  - [**Full report**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=R3Y8Q7cNLm56WvEb9gc9vF) showing data exploration and oversampling analysis

## 3. Running the code on your machine

### 3.1. Requisites

- anaconda
- Python 3.6 (Keras and TensorFlow don't work with Python 3.7)
- [Streamlit](https://streamlit.io/secret/docs/index.html)

This repo uses conda's virtual environment for Python 3.

#### Install (mini)conda if not yet installed:

For MacOS:
```shell
$ wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
$ chmod +x miniconda.sh
$ ./miniconda.sh -b
```

#### Create the virtual environment:

cd into the directory and create the ```debial-ml``` conda virtual environment from environment.yml
```shell
$ conda env create -f environment.yml
```

#### Activate the virtual environment:

```shell
$ source activate debias-ml
```

### 3.2. Running the code

As described above, there are four scripts which can be run to produce Streamlit reports.
- ```analysis_data.py```
- ```analysis_model.py```
- ```analysis_oversampling.py```
- ```analysis.py```

These can be all be run from the command line. To do this cd into the ```source``` directory and call,
```shell
$ python analysis_data.py
```
The scripts are listed in order of running time.

## 4. Data

Testing of this methodology was performed using census income data ([UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)):
- 32561 data points
- Target feature: ann_salary > $50K
- 14 features (in addition to the target feature) including race and gender
- 76% of the population earns less than $50K
- 67% of the population is male
- 85% of the population is white

### 4.1. File structure / data flow in the code

- The raw data files are saved in ```data/raw```
- The raw data is converted to csv format and saved as ```data/preprocessed/adult-data.csv```
- The input parameters are set manually in ```config/params.ini```
- After processing, the code saves a new csv file containing the processed data in ```data/processed/adult-data.csv```
- Parameters which are calculated in data processing and required for later calculations are written to the config file ```config/new_params.ini```

### 4.2. Running the code on a new data set

1. Save the csv file in the folder ```data/preprocessed/```
2. Edit the parameter values in the config file, ```config/params.ini```
3. Don't worry about overwriting the parameters for ```adult-data.csv```, a copy of the config file is saved as ```adult-data_params.ini```
4. Follow the instructions above for running the code

#### Notes:

- While efforts have been made to generalise, this code has not been tested on other datasets
- The Oversampler class is designed to remove bias from two sensitive features simultaneously

## 5. To Dos

- Increase test coverage
- Make it work for removing bias against a single sensitive feature
- Add a method to the Oversampler to output weights (rather than the oversampled data points)
- Check for and remove hard codes plot labels
- Write code to find features with bias and rank them
- Write bias_metrics class (follow sklearn)
- Design and implement infrastructure for stratified sampling on multiple dimensions
