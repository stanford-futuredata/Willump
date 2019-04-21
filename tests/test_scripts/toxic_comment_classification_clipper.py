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
    cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
    cascade_threshold = args.cascades

if args.batch is None:
    batch_size = 1
else:
    batch_size = args.batch


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


def willump_vectorizer_transform_caller(input_text):
    input_text = [str(x) for x in input_text]
    v2 = willump_execute_inner(vectorizer_transform, vt_source)
    preds = v2(input_text)
    return [str(pred) for pred in preds]


def python_vectorizer_transform_caller(input_text):
    input_text = [str(x) for x in input_text]
    preds = vectorizer_transform(input_text)
    return [str(pred) for pred in preds]


def vectorizer_transform(input_text):
    word_features = word_vectorizer.transform(input_text)
    char_features = char_vectorizer.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    preds = classifier.predict(combined_features)
    return preds


vt_source = inspect.getsource(vectorizer_transform)


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
    print("Stopping Clipper...")
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    sys.exit(0)


classifier = pickle.load(open(base_path + "model.pk", "rb"))
word_vectorizer, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))

if __name__ == '__main__':

    if args.disable:
        func_to_use = python_vectorizer_transform_caller
    else:
        func_to_use = willump_vectorizer_transform_caller

    signal.signal(signal.SIGINT, signal_handler)
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.start_clipper()
    clipper_conn.register_application("vectorizer-test", "strings", "default_pred", 15000000)
    python_deployer.deploy_python_closure(clipper_conn, name="vectorizer-model", version=1,
                                          input_type="strings", func=func_to_use,
                                          base_image='custom-model-image')
    clipper_conn.link_model_to_app("vectorizer-test", "vectorizer-model")
    time.sleep(2)

    train = pd.read_csv(base_path + 'train.csv').fillna(' ')
    train_text = list(train['comment_text'].values)

    class_name = "toxic"
    train_target = train[class_name]
    _, valid_text, _, valid_target = train_test_split(train_text, train_target, test_size=0.33, random_state=42)
    set_size = len(valid_text)

    i = 0
    try:
        while True:
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
