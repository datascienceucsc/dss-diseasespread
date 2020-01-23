# DengAI: Predicting Disease Spread

![A mosquito](http://www.pngall.com/wp-content/uploads/2016/05/Mosquito-High-Quality-PNG.png)

Repository for our work on the  DengAI: Predicting Disease Spread competition on Driven Data

## Installation

Start by following out usual [software setup guide](https://github.com/datascienceslugs/Useful-Documents/edit/master/setup_guide.md)

Branch this repository into a branch for your group, then clone it.

Next go to [drivendata.org](drivendata.org) and create an account. \

## Downloading original dataset
To get a fresh copy of the original dataset, go to `src/data` and run
```bash
python get_data.py
```

## Using automated hyperaparameter tuning

If you are familiar with `hyperopt`, define a search space yourself. Otherwise I provide a default one.
In your script in `src/models` use

```python
from xgb_optimization import optimize_xgb
best_params = optimize_xgb(X_train, y_train, num_iter)
```
Where `best_params` is a dictionary of the best parameters found, which you can then use as:
```python
from xgboost import XGBRegressor
xgb_model = XGBRegressor(**best_params)
```
If you define your search space yourself, use
```python
best_params = optimize_xgb(X_train, y_train, num_iter, hp_space = search_space)
```

## Saving your predictions

`starter.py` contains starter code that will show you how to save your predictions in the appropriate format for model stacking further down the line.

