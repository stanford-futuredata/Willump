# Original source: https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# Data files can be found on Kaggle:  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

import time
import pickle

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from adtracking_fraud_detection_util import *
from willump.evaluation.willump_executor import willump_execute

debug = 1

base_folder = "tests/test_resources/adtracking_fraud_detection/"


@willump_execute()
def process_input_and_train(input_df, input_y):
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
    clf = LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        num_leaves=7,
        max_depth=3,
        min_child_samples=100,
        max_bin=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_weight=0,
        scale_pos_weight=200,
    )
    clf = clf.fit(input_df, input_y, eval_metric="auc")
    return clf


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

    train_df, _, train_y, _ = train_test_split(train_df, train_y, test_size=0.1, random_state=42)
    num_rows = len(train_df)

    start = time.time()
    process_input_and_train(train_df, train_y)
    elapsed_time = time.time() - start
    print('First (Python) training in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    start = time.time()
    clf = process_input_and_train(train_df, train_y)
    elapsed_time = time.time() - start
    print('Second (Willump) training in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    pickle.dump(clf, open(base_folder + "model.pk", "wb"))
