import pandas as pd
import numpy as np
import time
import gc
import pickle
from sklearn import metrics
import willump.evaluation.willump_executor

# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32
# CLUSTER SIZES
UC_SIZE = 25
SC_SIZE = 25
USC_SIZE = 25
LATENT_USER_FEATURES = ['uf_' + str(i) for i in range(UF_SIZE)]
LATENT_SONG_FEATURES = ['sf_' + str(i) for i in range(SF_SIZE)]
UC_FEATURES = [
    'uc_played',
    'uc_played_pos',
    'uc_played_rel',
    'uc_played_rel_global',
    'uc_played_pos_rel',
    'uc_played_ratio'
]
SC_FEATURES = [
    'sc_played',
    'sc_played_pos',
    'sc_played_rel',
    'sc_played_rel_global',
    'sc_played_ratio'
]

AC_FEATURES = [
    'ac_played',
    'ac_played_pos',
    'ac_played_rel',
    'ac_played_rel_global',
    'ac_played_ratio'
]
FEATURES = LATENT_SONG_FEATURES + LATENT_USER_FEATURES
FEATURES = FEATURES + SC_FEATURES
FEATURES = FEATURES + UC_FEATURES
FEATURES = FEATURES + AC_FEATURES


def auc_score(y_valid, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def load_combi_prep(folder='data_new/', split=None):
    print('LOAD COMBI')

    start = time.time()

    name = 'combi_extra' + ('.' + str(split) if split is not None else '') + '.pkl'
    combi = pd.read_pickle(folder + name)

    print('loaded in: ', (time.time() - start))

    return combi


model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))


@willump.evaluation.willump_executor.willump_execute(batch=False, num_workers=0)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col):
    c_one_combi = combi.merge(cluster_one, how='left', on=join_col_cluster_one)
    c_two_combi = c_one_combi.merge(cluster_two, how='left', on=join_col_cluster_two)
    c_three_combi = c_two_combi.merge(cluster_three, how='left', on=join_col_cluster_three)
    als_one_combi = c_three_combi.merge(features_one, how='left', on=join_col_one)
    als_two_combi = als_one_combi.merge(features_two, how='left', on=join_col_two)
    uc_combi = als_two_combi.merge(uc_features, how='left', on=uc_join_col)
    sc_combi = uc_combi.merge(sc_features, how='left', on=sc_join_col)
    ac_combi = sc_combi.merge(ac_features, how='left', on=ac_join_col)
    combi_features = ac_combi[FEATURES]
    preds = model.predict(combi_features)
    return preds


def load_als_dataframe(folder, size, user, artist):
    if user:
        if artist:
            name = 'user2'
            key = 'uf2_'
        else:
            name = 'user'
            key = 'uf_'
    else:
        if artist:
            name = 'artist'
            key = 'af_'
        else:
            name = 'song'
            key = 'sf_'
    csv_name = folder + 'als' + '_' + name + '_features' + '.{}.csv'.format(size)
    features = pd.read_csv(csv_name)

    for i in range(size):
        features[key + str(i)] = features[key + str(i)].astype(np.float32)
    oncol = 'msno' if user else 'song_id' if not artist else 'artist_name'
    return features, oncol


def scol_features(folder, col, prefix):
    csv_name = folder + prefix + 'cluster' + '_' + col + '_features' + '.csv'
    tmp = pd.read_csv(csv_name)
    return tmp, col


def add_cluster(folder, col, size, overlap=True, positive=True, content=False):
    name = 'cluster_' + col
    file_name = 'alsclusterEMB32_' + col
    if content:
        file_name = 'content_' + file_name
    if overlap:
        file_name += '_ol'
    if not positive:
        file_name += '_nopos'

    # cluster = pd.read_csv( folder + 'content_' + name +'.{}.csv'.format(size) )
    cluster = pd.read_csv(folder + file_name + '.{}.csv'.format(size))

    cluster[name + '_' + str(size)] = cluster.cluster_id
    del cluster['cluster_id']

    return cluster, col


def add_als(folder, combi):
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)
    cluster_one, join_col_cluster_one = add_cluster(folder, col='msno', size=UC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_two, join_col_cluster_two = add_cluster(folder, col='song_id', size=SC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_three, join_col_cluster_three = add_cluster(folder, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features(folder, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features(folder, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features(folder, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')

    num_rows = len(combi)
    compile_one = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col)
    compile_two = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col)
    compile_three = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                             join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                             join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                             ac_join_col)
    set_size = len(combi)
    entry_list = []
    for i in range(set_size):
        entry_list.append(combi.iloc[i:i + 1])
    print("Copies created")
    start = time.time()
    for entry in entry_list:
        pred = do_merge(entry, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                     join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                     join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                     ac_join_col)
    elapsed_time = time.time() - start

    print('Latent feature join in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    return combi


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])

    # partition data by time
    # combi['time_bin10'] = pd.cut(combi['time'], 10, labels=range(10))
    # combi['time_bin5'] = pd.cut(combi['time'], 5, labels=range(5))

    # add latent features
    combi = add_als(folder, combi)


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
