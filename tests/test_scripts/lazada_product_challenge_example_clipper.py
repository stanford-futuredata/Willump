from __future__ import print_function

import argparse
import json
import pickle
import signal
import sys
import cloudpickle
import time
import inspect
from datetime import datetime

import pandas as pd
import requests
import scipy.sparse
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
from sklearn.model_selection import train_test_split

from willump.evaluation.willump_executor import willump_execute_inner

base_path = "tests/test_resources/toxic_comment_classification/"
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
parser.add_argument("-b", "--batch", type=int, help="Batch Size")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    assert (not args.disable)
    cascades = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "rb"))
    cascade_threshold = args.cascades

if args.batch is None:
    batch_size = 1
else:
    batch_size = args.batch

latencies = []


def predict(addr, x, batch=False):
    url = "http://%s/vectorizer-test/predict" % addr

    if batch:
        # print(x)
        req_json = json.dumps({'input_batch': x})
    else:
        # print(x)
        req_json = json.dumps({'input': x})

    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))
    latencies.append(latency)


def willump_vectorizer_transform_caller(input_text):
    input_text = [str(x) for x in input_text]
    v2 = willump_execute_inner(vectorizer_transform, vt_source,
                               eval_cascades=cascades, cascade_threshold=cascade_threshold)
    preds = v2(input_text)
    return [str(pred) for pred in preds]


def python_vectorizer_transform_caller(input_text):
    input_text = [str(x) for x in input_text]
    preds = vectorizer_transform(input_text)
    return [str(pred) for pred in preds]


def vectorizer_transform(input_text):
    transformed_result = title_vectorizer.transform(input_text)
    color_result = color_vectorizer.transform(input_text)
    brand_result = brand_vectorizer.transform(input_text)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    predictions = model.predict(combined_result)
    return predictions


vt_source = inspect.getsource(vectorizer_transform)


model = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "rb"))
title_vectorizer, color_vectorizer, brand_vectorizer = pickle.load(
    open("tests/test_resources/lazada_challenge_features/lazada_vectorizers.pk", "rb"))

if __name__ == '__main__':

    if args.disable:
        func_to_use = python_vectorizer_transform_caller
    else:
        func_to_use = willump_vectorizer_transform_caller

    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.start_clipper()
    clipper_conn.register_application("vectorizer-test", "strings", "default_pred", 15000000)
    python_deployer.deploy_python_closure(clipper_conn, name="vectorizer-model", version=1,
                                          input_type="strings", func=func_to_use,
                                          base_image='custom-model-image')
    clipper_conn.link_model_to_app("vectorizer-test", "vectorizer-model")
    time.sleep(2)

    train = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])
    train_text = list(train['title'].values)

    _, valid_text = train_test_split(train_text, test_size=0.33, random_state=42)
    set_size = len(valid_text)

    i = 0
    try:
        for _ in range(200):
            if batch_size > 1:
                predict(
                    clipper_conn.get_query_addr(),
                    valid_text[i:i + batch_size],
                    batch=True)
            else:
                predict(clipper_conn.get_query_addr(), valid_text[i])
            i += batch_size
            time.sleep(0.2)
    except Exception as e:
        clipper_conn.stop_all()

    pickle.dump(latencies[20:], open("latencies.pk", "wb"))
    clipper_conn.stop_all()