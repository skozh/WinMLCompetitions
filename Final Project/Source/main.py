#!/usr/bin/env python3
# coding: utf-8

# # Final Project: Sales Prediction

# ## Import Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sklearn
import scipy.sparse 
from itertools import product
import gc
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# ## Load Data

sales = pd.read_csv('./Data/sales_train.csv')
shops = pd.read_csv('./Data/shops.csv')
items = pd.read_csv('./Data/items.csv')
item_cats = pd.read_csv('./Data/item_categories.csv')
test = pd.read_csv('./Data/test.csv')


###  Load functions
from funs import downcast_dtypes, rmse,  get_feature_matrix, clip20, clip40


def main():
    list_lags = [1, 2, 3, 4]
    date_block_threshold = 6
    sales_for_modelling = sales[sales.item_id.isin(test.item_id)]
    [all_data, to_drop_cols]  = get_feature_matrix(sales_for_modelling, test, items, list_lags, date_block_threshold)

    all_data = downcast_dtypes(all_data)
    # Keep test data separate
    sub_data = all_data[all_data.date_block_num==34].fillna(0)
    all_data = all_data[all_data.date_block_num<34].fillna(0)

    # Keeping months 30-33 data for validation
    dates = all_data['date_block_num']
    boolean_test = (dates.isin([30,31,32,33])) # & (boolean)
    boolean_train = ~boolean_test
    dates_train = dates[boolean_train]
    dates_val  = dates[boolean_test]

    X_train = all_data.loc[boolean_train].drop(to_drop_cols, axis=1)
    X_val =  all_data.loc[boolean_test].drop(to_drop_cols, axis=1)
    y_train = all_data.loc[boolean_train, 'target'].values
    y_val =  all_data.loc[boolean_test, 'target'].values

    X = X_train.append(X_val)
    y = np.concatenate([y_train, y_val])

    rf = RandomForestRegressor(bootstrap=0.7, criterion='mse', max_depth=12,
            max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=4, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

    rf.fit(X, clip40(y))



    pred_rf_val = clip20(rf.predict(X_val.fillna(0)))


    test_pred = rf.predict(sub_data.drop(to_drop_cols, axis=1).fillna(0))


    try:
        os.unlink('./Output/submission.csv')
    except OSError:
        pass


    predictions = pd.DataFrame()
    predictions['shop_id'] = test.shop_id
    predictions['item_id'] = test.item_id
    predictions['item_cnt_month'] = test_pred
    submision = test[['ID', 'shop_id', 'item_id']].merge(predictions, on=['shop_id', 'item_id'], how='left').fillna(0)
    submision[['ID', 'item_cnt_month']].to_csv('./Output/submission.csv',index=False)
    print("Output csv generated !!!")


if __name__ == "__main__":
    main()
