# Original source: https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# Data files can be found on Kaggle:  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

import gc
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

debug = 1

base_folder = "tests/test_resources/adtracking_fraud_detection/"


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.2,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics + ":", evals_results['valid'][metrics][bst1.best_iteration - 1])

    return bst1, bst1.best_iteration


if __name__ == "__main__":
    max_rows = 10000000
    if debug:
        train_start_point = 0
        train_end_point = 100000
    else:
        train_start_point = 0
        train_end_point = max_rows

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    print('loading train data...', train_start_point, train_end_point)
    train_df = pd.read_csv(base_folder + "train.csv", parse_dates=['click_time'], skiprows=range(1, train_start_point),
                           nrows=train_end_point - train_start_point,
                           dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    len_train = len(train_df)
    gc.collect()

    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

    print('doing nextClick')
    nextClick_filename = base_folder + 'nextClick_%d_%d.csv' % (train_start_point, train_end_point)
    if os.path.exists(nextClick_filename):
        print('loading from save file')
        nextClick = pd.read_csv(nextClick_filename).values
    else:
        D = 2 ** 26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df[
            'device'].astype(str) + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer = np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime'] = train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category] - t)
            click_buffer[category] = t
        del click_buffer
        nextClick = list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(nextClick).to_csv(nextClick_filename, index=False)

    train_df['nextClick'] = nextClick
    train_df['nextClick_shift'] = pd.DataFrame(nextClick).shift(+1).values
    train_df = train_df.drop(columns=["click_time"])
    del nextClick
    gc.collect()

    print("Adding X Features")
    selected_columns = ['ip', 'channel']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X0'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['ip', 'device', 'os', 'app']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[selected_columns[-1]].cumcount()
    train_df['X1'] = gp.values

    selected_columns = ['ip', 'day', 'hour']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X2'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['ip', 'app']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X3'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['ip', 'app', 'os']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X4'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['ip', 'device']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X5'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['app', 'channel']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X6'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    selected_columns = ['ip', 'os']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[selected_columns[-1]].cumcount()
    train_df['X7'] = gp.values

    selected_columns = ['ip', 'device', 'os', 'app']
    gp = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X8'})
    train_df = train_df.merge(gp, on=selected_columns[0:-1], how='left')

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])[
        ['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(
        index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[
        ['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[
        ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()

    target = 'is_attributed'
    predictors = ['nextClick', 'nextClick_shift', 'app', 'device', 'os', 'channel', 'hour', 'day', 'ip_tcount',
                  'ip_tchan_count', 'ip_app_count', 'ip_app_os_count', 'ip_app_os_var', 'ip_app_channel_var_day',
                  'ip_app_channel_mean_hour', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    print('Predictors', predictors)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }
    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              train_df,
                                              val_df,
                                              predictors,
                                              target,
                                              objective='binary',
                                              metrics='auc',
                                              early_stopping_rounds=30,
                                              num_boost_round=1000,
                                              categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    print("done...")
