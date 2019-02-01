import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import os
import willump.evaluation.willump_executor
from wsdm_features_list import FEATURES

# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32
# CLUSTER SIZES
UC_SIZE = 25
SC_SIZE = 25
USC_SIZE = 25


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


# @willump.evaluation.willump_executor.willump_execute(batch=True)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col, us_features, us_col, ss_features, ss_col, as_features,
             as_col, gs_features, gs_col, cs_features, cs_col, ages_features, ages_col, ls_features, ls_col,
             gender_features, gender_col, comps_features, comps_col, lyrs_features, lyrs_col, snames_features,
             snames_col, stabs_features, stabs_col, stypes_features, stypes_col, regs_features, regs_col):
    c_one_combi = combi.merge(cluster_one, how='left', on=join_col_cluster_one)
    c_two_combi = c_one_combi.merge(cluster_two, how='left', on=join_col_cluster_two)
    c_three_combi = c_two_combi.merge(cluster_three, how='left', on=join_col_cluster_three)
    als_one_combi = c_three_combi.merge(features_one, how='left', on=join_col_one)
    als_two_combi = als_one_combi.merge(features_two, how='left', on=join_col_two)
    uc_combi = als_two_combi.merge(uc_features, how='left', on=uc_join_col)
    sc_combi = uc_combi.merge(sc_features, how='left', on=sc_join_col)
    ac_combi = sc_combi.merge(ac_features, how='left', on=ac_join_col)
    us_combi = ac_combi.merge(us_features, how='left', on=us_col)
    ss_combi = us_combi.merge(ss_features, how='left', on=ss_col)
    as_combi = ss_combi.merge(as_features, how='left', on=as_col)
    gs_combi = as_combi.merge(gs_features, how='left', on=gs_col)
    cs_combi = gs_combi.merge(cs_features, how='left', on=cs_col)
    ages_combi = cs_combi.merge(ages_features, how='left', on=ages_col)
    ls_combi = ages_combi.merge(ls_features, how='left', on=ls_col)
    gender_combi = ls_combi.merge(gender_features, how='left', on=gender_col)
    comps_combi = gender_combi.merge(comps_features, how='left', on=comps_col)
    lyrs_combi = comps_combi.merge(lyrs_features, how='left', on=lyrs_col)
    snames_combi = lyrs_combi.merge(snames_features, how='left', on=snames_col)
    stabs_combi = snames_combi.merge(stabs_features, how='left', on=stabs_col)
    stypes_combi = stabs_combi.merge(stypes_features, how='left', on=stypes_col)
    regs_combi = stypes_combi.merge(regs_features, how='left', on=regs_col)
    return regs_combi


def scol_features(folder, combi, col, prefix):
    tmp = pd.DataFrame()
    group = combi.groupby([col])
    group_pos = combi[combi.target == 1].groupby(col)
    tmp[prefix + 'played'] = group.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = group_pos.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = tmp[prefix + 'played_pos'].fillna(0)
    tmp[prefix + 'played_rel'] = (tmp[prefix + 'played'] / tmp[prefix + 'played'].max()).astype(np.float32)
    tmp[prefix + 'played_rel_global'] = (tmp[prefix + 'played'] / len(combi)).astype(np.float32)
    tmp[prefix + 'played_pos_rel'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played_pos'].max()).astype(np.float32)
    tmp[prefix + 'played_ratio'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played']).astype(np.float32)
    tmp[col] = tmp.index
    csv_name = folder + prefix + 'cluster' + '_' + col + '_features' + '.csv'
    tmp.to_csv(csv_name)
    return tmp, col


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


def add_cluster(folder, col, size, overlap=True, positive=True, content=False):
    global total_join_time
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


def add_features(folder, combi):
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)
    cluster_one, join_col_cluster_one = add_cluster(folder, col='msno', size=UC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_two, join_col_cluster_two = add_cluster(folder, col='song_id', size=SC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_three, join_col_cluster_three = add_cluster(folder, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    uc_combi = combi.merge(cluster_one, how='left', on=join_col_cluster_one)
    uc_combi = uc_combi.merge(cluster_two, how='left', on=join_col_cluster_two)
    uc_combi = uc_combi.merge(cluster_three, how='left', on=join_col_cluster_three)
    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features(folder, uc_combi, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features(folder, uc_combi, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features(folder, uc_combi, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    # USER FEATURES
    us_features, us_col = scol_features(folder, combi, 'msno', 'u_')
    # SONG FEATURES
    ss_features, ss_col = scol_features(folder, combi, 'song_id', 's_')
    # ARTIST FEATURES
    as_features, as_col = scol_features(folder, combi, 'artist_name', 'a_')
    # GENRE FEATURES
    gs_features, gs_col = scol_features(folder, combi, 'genre_max', 'gmax_')
    # CITY FEATURES
    cs_feature, cs_col = scol_features(folder, combi, 'city', 'c_')
    # AGE FEATURES
    ages_features, ages_col = scol_features(folder, combi, 'bd', 'age_')
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features(folder, combi, 'language', 'lang_')
    # GENDER FEATURES
    gender_features, gender_col = scol_features(folder, combi, 'gender', 'gen_')
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features(folder, combi, 'composer', 'comp_')
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features(folder, combi, 'lyricist', 'ly_')
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features(folder, combi, 'source_screen_name', 'sn_')
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features(folder, combi, 'source_system_tab', 'sst_')
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features(folder, combi, 'source_type', 'st_')
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features(folder, combi, 'registered_via', 'rv_')

    num_rows = len(combi)
    compile_one = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_feature, cs_col,
                           ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                           composer_col,
                           lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                           stypes_col, regs_features, regs_col)
    compile_two = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_feature, cs_col,
                           ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                           composer_col,
                           lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                           stypes_col, regs_features, regs_col)
    compile_three = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                             join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                             join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                             ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                             gs_col, cs_feature, cs_col,
                             ages_features, ages_col, ls_features, ls_col, gender_features, gender_col,
                             composer_features,
                             composer_col,
                             lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                             stypes_col, regs_features, regs_col)

    start = time.time()
    combi = do_merge(combi, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                     join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                     join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                     ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                     gs_col, cs_feature, cs_col,
                     ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                     composer_col,
                     lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                     stypes_col, regs_features, regs_col)
    elapsed_time = time.time() - start

    print('Feature join in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    return combi


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])

    # partition data by time
    # combi['time_bin10'] = pd.cut(combi['time'], 10, labels=range(10))
    # combi['time_bin5'] = pd.cut(combi['time'], 5, labels=range(5))

    # add latent features
    combi = add_features(folder, combi)
    model = LogisticRegression()
    y = combi["target"].values
    train_data = combi[FEATURES].values
    model.fit(train_data, y)

    y_pred = model.predict(train_data)

    print("Train AUC: %f" % auc_score(y, y_pred))
    pickle.dump(model, open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "wb"))


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
