# Original source: https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# Data files can be found on Kaggle:  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

import pickle
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from adtracking_fraud_detection_util import *
from willump.evaluation.willump_executor import willump_execute

debug = 1

base_folder = "tests/test_resources/adtracking_fraud_detection/"


@willump_execute()
def process_input_and_predict(input_df):
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
    preds = clf.predict(input_df)
    return preds


if __name__ == "__main__":
    clf = pickle.load(open(base_folder + "model.pk", "rb"))

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

    _, valid_df, _, valid_y = train_test_split(train_df, train_y, test_size=0.1, random_state=42)
    del train_df, train_y

    num_rows = len(valid_df)

    mini_df = valid_df[0:2]
    process_input_and_predict(mini_df)
    process_input_and_predict(mini_df)
    start = time.time()
    y_preds = process_input_and_predict(valid_df)
    elapsed_time = time.time() - start
    print('Prediction in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    print("Validation ROC-AUC Score: %f" % roc_auc_score(valid_y, y_preds))
