import gc
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

categorical_indices = None
random_state = None


def willump_train_function(X, y):
    model = LGBMClassifier(
        n_jobs=1,
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
        random_state=random_state
    )
    model = model.fit(X, y, eval_metric='auc', categorical_feature=categorical_indices)
    return model


def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    else:
        return model.predict(X)


def willump_predict_proba_function(model, X):
    return model.predict_proba(X)[:, 1]


def willump_score_function(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)


def gen_aggregate_statistics_tables(train_df, base_folder, train_start_point, train_end_point, debug):
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

    gc.collect()

    selected_columns = ['ip', 'channel']
    X_ip_channel = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X0'})
    X_ip_channel_jc = selected_columns[0:-1]

    selected_columns = ['ip', 'device', 'os', 'app']
    X_ip_device_os_app = train_df[selected_columns].groupby(by=selected_columns[0:-1])[selected_columns[-1]].cumcount()

    selected_columns = ['ip', 'day', 'hour']
    X_ip_day_hour = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X2'})
    X_ip_day_hour_jc = selected_columns[0:-1]

    selected_columns = ['ip', 'app']
    X_ip_app = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X3'})
    X_ip_app_jc = selected_columns[0:-1]

    selected_columns = ['ip', 'app', 'os']
    X_ip_app_os = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X4'})
    X_ip_app_os_jc = selected_columns[0:-1]

    selected_columns = ['ip', 'device']
    X_ip_device = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X5'})
    X_ip_device_jc = selected_columns[0:-1]

    selected_columns = ['app', 'channel']
    X_app_channel = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X6'})
    X_app_channel_jc = selected_columns[0:-1]

    selected_columns = ['ip', 'os']
    X_ip_os = train_df[selected_columns].groupby(by=selected_columns[0:-1])[selected_columns[-1]].cumcount()

    selected_columns = ['ip', 'device', 'os', 'app']
    X_ip_device_app_os = train_df[selected_columns].groupby(by=selected_columns[0:-1])[
        selected_columns[-1]].nunique().reset_index(). \
        rename(index=str, columns={selected_columns[-1]: 'X8'})
    X_ip_device_app_os_jc = selected_columns[0:-1]

    ip_day_hour = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    ip_day_hour_jc = ['ip', 'day', 'hour']

    ip_app = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'ip_app_count'})
    ip_app_jc = ['ip', 'app']

    ip_app_os = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    ip_app_os_jc = ['ip', 'app', 'os']

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    ip_day_hour_channel = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])[
        ['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    ip_day_hour_channel_jc = ['ip', 'day', 'channel']

    ip_app_os_hour = train_df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[
        ['hour']].var().reset_index().rename(
        index=str, columns={'hour': 'ip_app_os_var'})
    ip_app_os_hour_jc = ['ip', 'app', 'os']

    ip_app_channel_var_day = train_df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[
        ['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    ip_app_channel_var_day_jc = ['ip', 'app', 'channel']

    ip_app_chl_mean_hour = train_df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[
        ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    ip_app_chl_mean_hour_jc = ['ip', 'app', 'channel']

    return X_ip_channel, X_ip_channel_jc, X_ip_day_hour, X_ip_day_hour_jc, X_ip_app, X_ip_app_jc, \
           X_ip_app_os, X_ip_app_os_jc, \
           X_ip_device, X_ip_device_jc, X_app_channel, X_app_channel_jc, X_ip_device_app_os, X_ip_device_app_os_jc, \
           ip_app_os, ip_app_os_jc, ip_day_hour, ip_day_hour_jc, ip_app, ip_app_jc, ip_day_hour_channel, \
           ip_day_hour_channel_jc, ip_app_channel_var_day, ip_app_channel_var_day_jc, ip_app_os_hour, \
           ip_app_os_hour_jc, ip_app_chl_mean_hour, ip_app_chl_mean_hour_jc, \
           nextClick, pd.DataFrame(nextClick).shift(+1).values, X_ip_device_os_app.values, X_ip_os.values
