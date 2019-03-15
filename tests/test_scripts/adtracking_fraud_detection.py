# Original source: https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# Data files can be found on Kaggle:  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

import time

import lightgbm as lgb
from sklearn.model_selection import train_test_split

from adtracking_fraud_detection_util import *
from willump.evaluation.willump_executor import willump_execute

debug = 1

base_folder = "tests/test_resources/adtracking_fraud_detection/"


def lgb_modelfit_nocv(params, dtrain, dvalid, train_y, valid_y, objective='binary', metrics='auc',
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

    xgtrain = lgb.Dataset(dtrain.values, label=train_y,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid.values, label=valid_y,
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


@willump_execute()
def process_input(input_df):
    input_df = input_df.merge(X_ip_channel, how='left', on=X_ip_channel_jc)
    input_df = input_df.merge(X_ip_day_hour, how='left', on=X_ip_day_hour_jc)
    input_df = input_df.merge(X_ip_app, how='left', on=X_ip_app_jc)
    input_df = input_df.merge(X_ip_app_os, how='left', on=X_ip_app_os_jc)
    input_df = input_df.merge(X_ip_device, how='left', on=X_ip_device_jc)
    input_df = input_df.merge(X_app_channel, how='left', on=X_app_channel_jc)
    input_df = input_df.merge(X_ip_device_app_os, how='left', on=X_ip_device_app_os_jc)
    input_df = input_df.merge(ip_app_os, how='left', on=ip_app_os_jc)
    input_df = input_df.merge(ip_day_hour, how='left', on=ip_day_hour_jc)
    input_df = input_df.merge(ip_app, how='left', on=ip_app_jc)
    input_df = input_df.merge(ip_day_hour_channel, how='left', on=ip_day_hour_channel_jc)
    input_df = input_df.merge(ip_app_channel_var_day, how='left', on=ip_app_channel_var_day_jc)
    input_df = input_df.merge(ip_app_os_hour, how='left', on=ip_app_os_hour_jc)
    input_df = input_df.merge(ip_app_chl_mean_hour, how='left', on=ip_app_chl_mean_hour_jc)
    input_df = input_df[predictors]
    return input_df


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

    (train_df, X_ip_channel, X_ip_channel_jc, X_ip_day_hour, X_ip_day_hour_jc, X_ip_app, X_ip_app_jc,
    X_ip_app_os, X_ip_app_os_jc,
    X_ip_device, X_ip_device_jc, X_app_channel, X_app_channel_jc, X_ip_device_app_os, X_ip_device_app_os_jc,
    ip_app_os, ip_app_os_jc, ip_day_hour, ip_day_hour_jc, ip_app, ip_app_jc, ip_day_hour_channel,
    ip_day_hour_channel_jc, ip_app_channel_var_day, ip_app_channel_var_day_jc, ip_app_os_hour,
    ip_app_os_hour_jc, ip_app_chl_mean_hour, ip_app_chl_mean_hour_jc) = \
        gen_aggregate_statistics_tables(train_df,
                                        base_folder,
                                        train_start_point,
                                        train_end_point,
                                        debug)

    print("Train Dataframe Information: ")
    train_df.info()

    target = 'is_attributed'
    predictors = ['nextClick', 'nextClick_shift', 'app', 'device', 'os', 'channel', 'hour', 'day', 'ip_tcount',
                  'ip_tchan_count', 'ip_app_count', 'ip_app_os_count', 'ip_app_os_var', 'ip_app_channel_var_day',
                  'ip_app_channel_mean_hour', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    train_y = train_df[target].values
    process_input(train_df)
    train_df = process_input(train_df)

    train_df, val_df, train_y, val_y = train_test_split(train_df, train_y, test_size=0.1, random_state=42)

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
                                              train_y,
                                              val_y,
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
