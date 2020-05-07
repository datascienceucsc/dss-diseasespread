# DengAI: Predicting Disease Spread
*Anders Poirel, Julian Lehrer*

![A mosquito](http://www.pngall.com/wp-content/uploads/2016/05/Mosquito-High-Quality-PNG.png)

Repository for our work on the  DengAI: Predicting Disease Spread competition on Driven Data

## Installation

You'll need to have `conda` installed.
Clone this repo.
Next go to [drivendata.org](drivendata.org) and create an account if you want to see your leaderboard performance.

### Setting up your environment

Run
```bash
conda env create -f environment.yml
```
If you already have the environment set up and want to switch to it, run,
```bash
conda activate dss-dengueai
```

## Downloading original dataset
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

CLassical linear models:
- [notebooks/apoirels-models-01.ipynb](notebooks/apoirel-models-01.ipynb) 01 through 04

FBProphet models:
- [notebooks/apoirels-prophet-01.ipynb](notebooks/apoirel-models-01.ipynb) 01 through 03

## Results

Current leaderboard position : 1067 (top 12%)
