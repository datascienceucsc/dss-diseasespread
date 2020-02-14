# DengAI: Predicting Disease Spread
*Anders Poirel, Julien Lehrer*

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

## Results

Current leaderboard position : 2191 (top 26%)