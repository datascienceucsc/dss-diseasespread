# DengAI: Predicting Disease Spread
*Anders Poirel, Julian Lehrer*

Repository for our work on the  [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) competition on Driven Data

- write-up [link](https://anderspoirel.me/reports/lessons-denguai/)

## Installation

*Requirements*:
- `conda`

### Setting up your environment

Run
```bash
conda env create -f environment.yml
```
Activate the environment
```bash
conda activate dss-dengueai
```

### Downloading original dataset
To get a fresh copy of the original dataset, go to `src/data` and run
```bash
python get_data.py
```

## Work

Data exploration:
- Anders [notebooks/apoirel-exploration-01.ipynb](notebooks/apoirel-exploration-01.ipynb)
- Julian [notebooks/jlehrer-exploration-01.ipynb](notebooks/jlehrer-exploration-01.ipynb)

Gradient boosted models:
- [notebooks/jlehrer-models-01.ipynb](notebooks/jlehrer-models-01.ipynb)

Baselines using linear models:
- [notebooks/apoirel-models-01.ipynb](notebooks/apoirel-models-01.ipynb) 01 through 04

Linear model with time-lagged features
- [notebooks/apoirel-improved-fe-02.ipynb](notebooks/apoirel-improved-fe-02.ipynb)

Linear model with time-lagged features and predictions using target values with time lag 1
- [notebooks/apoirel-models-05.ipynb](notebooks/apoirel-improved-fe-02.ipynb)

FBProphet experiments with more or less models:
- [notebooks/apoirels-prophet-01.ipynb](notebooks/apoirel-prophet-01.ipynb) 01 through 03

## 🏆 Results 

Current leaderboard position : 692 (top 7.6%)
Previous submission position: 1100 (top 12%)
