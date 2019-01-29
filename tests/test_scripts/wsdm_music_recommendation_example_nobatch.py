import pandas as pd
import numpy as np
import time
import gc
import pickle
import willump.evaluation.willump_executor

# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32
LATENT_USER_FEATURES = ['uf_' + str(i) for i in range(UF_SIZE)]
LATENT_SONG_FEATURES = ['sf_' + str(i) for i in range(SF_SIZE)]
FEATURES = LATENT_SONG_FEATURES + LATENT_USER_FEATURES


def load_combi_prep(folder='data_new/', split=None):
    print('LOAD COMBI')

    start = time.time()

    name = 'combi_extra' + ('.' + str(split) if split is not None else '') + '.pkl'
    combi = pd.read_pickle(folder + name)

    print('loaded in: ', (time.time() - start))

    return combi


model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))


@willump.evaluation.willump_executor.willump_execute(batch=False, num_workers=1)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two):
    one_combi = combi.merge(features_one, how='left', on=join_col_one)
    two_combi = one_combi.merge(features_two, how='left', on=join_col_two)
    two_combi_features = two_combi[FEATURES]
    preds = model.predict(two_combi_features)
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


def add_als(folder, combi):
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)

    num_rows = len(combi)
    compile_one = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf)
    compile_two = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf)
    compile_three = do_merge(combi.iloc[0:1].copy(), features_uf, join_col_uf, features_sf, join_col_sf)
    set_size = len(combi)
    entry_list = []
    for i in range(set_size):
        entry_list.append(combi.iloc[i:i + 1])
    print("Copies created")
    start = time.time()
    for entry in entry_list:
        pred = do_merge(entry, features_uf, join_col_uf, features_sf, join_col_sf)
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
