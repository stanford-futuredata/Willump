import pandas as pd
import numpy as np
import time
import gc
import willump.evaluation.willump_executor

# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32


def load_combi_prep(folder='data_new/', split=None):
    print('LOAD COMBI')

    start = time.time()

    name = 'combi_extra' + ('.' + str(split) if split is not None else '') + '.pkl'
    combi = pd.read_pickle(folder + name)

    print('loaded in: ', (time.time() - start))

    return combi


@willump.evaluation.willump_executor.willump_execute
def do_merge(combi, features, how, on):
    combi = combi.merge(features, how='inner', on='msno')
    return combi


def add_als(folder, combi, size, user=True, pos=False, postfix='', artist=False):
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
    csv_name = folder + 'als' + postfix + '_' + name + '_features' + ('_pos' if pos else '') + '.{}.csv'.format(size)
    features = pd.read_csv(csv_name)

    for i in range(size):
        features[key + str(i)] = features[key + str(i)].astype(np.float32)

    num_rows = len(combi)
    oncol = 'msno' if user else 'song_id' if not artist else 'artist_name'
    compile_one = do_merge(combi.copy(), features, 'inner', oncol)
    compile_two = do_merge(combi.copy(), features, 'inner', oncol)
    compile_three = do_merge(combi.copy(), features, 'inner', oncol)

    start = time.time()
    combi = do_merge(combi, features, 'inner', oncol)
    elapsed_time = time.time() - start

    del features
    gc.collect()
    print('Latent feature %s join in %f seconds rows %d throughput %f: ' % (
        name, elapsed_time, num_rows, num_rows / elapsed_time))

    return combi


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)

    # partition data by time
    combi['time_bin10'] = pd.cut(combi['time'], 10, labels=range(10))
    combi['time_bin5'] = pd.cut(combi['time'], 5, labels=range(5))

    # add latent features
    combi = add_als(folder, combi, UF_SIZE, user=True, postfix='', artist=False)


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
