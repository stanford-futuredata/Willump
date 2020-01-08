# Benchmarks

This document describes how to reproduce the results in Figures 6, 7, and 8 of
the [Willump paper](https://arxiv.org/pdf/1906.01974.pdf).  It assumes Willump has already
been installed following the directions
[here](https://github.com/stanford-futuredata/Willump/blob/master/README.md).

This repository contains scripts to reproduce six of Willump's seven benchmarks.  To reproduce
the seventh, Purchase, see the separate [willump-dfs repository](https://github.com/stanford-futuredata/willump-dfs), 
which contains Willump's [Featuretools](https://www.featuretools.com/) integration.

Datasets for three of the six benchmarks (Toxic, Product, and Music) are already included in this
repository.  The three remaining datasets can be downloaded from Kaggle and placed in their
respective folders in tests/test_resources.
The dataset for Credit is [here](https://www.kaggle.com/c/home-credit-default-risk/data)
and files go in the mercari_price_suggestion folder,
the dataset for Price is [here](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)
and files go in the home_credit_default_risk folder,
and the dataset for Instant is [here](https://www.kaggle.com/c/instant-gratification/data)
and files go in the instant_gratification folder.

Note that some benchmarks load data into and then query data from a remote Redis server.

We will now describe how to reproduce each of the benchmarks.

### Figure 6

To reproduce the offline batch experiments described in Figure 6, first train a model for each benchmark:

    python3 tests/benchmark_scripts/toxic_train.py
    python3 tests/benchmark_scripts/product_train.py
    python3 tests/benchmark_scripts/music_train.py
    python3 tests/benchmark_scripts/credit_train.py
    python3 tests/benchmark_scripts/price_train.py
    python3 tests/benchmark_scripts/instant_train.py
    
To measure each benchmark's unoptimized runtime, run each with Willump disabled:

    python3 tests/benchmark_scripts/toxic_batch.py -d
    python3 tests/benchmark_scripts/product_batch.py -d
    python3 tests/benchmark_scripts/music_batch.py -d
    python3 tests/benchmark_scripts/music_remote_batch.py -d -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/credit_batch.py -d
    python3 tests/benchmark_scripts/price_batch.py -d
    python3 tests/benchmark_scripts/instant_batch.py -d
    
To measure the effects of Willump's compilation optimizations on runtime for compilable benchmarks, run:

    python3 tests/benchmark_scripts/toxic_batch.py
    python3 tests/benchmark_scripts/product_batch.py
    python3 tests/benchmark_scripts/music_batch.py
    python3 tests/benchmark_scripts/credit_batch.py
    python3 tests/benchmark_scripts/price_batch.py
    
To measure the effects of Willump's cascades optimization on runtime for classification benchmarks, run:

    python3 tests/benchmark_scripts/toxic_batch.py -c
    python3 tests/benchmark_scripts/product_batch.py -c
    python3 tests/benchmark_scripts/music_batch.py -c
    python3 tests/benchmark_scripts/music_remote_batch.py -c -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/instant_batch.py -c

Each benchmark reports its throughput.
For each benchmark, throughput trends should resemble those in Figure 6.

### Figure 7

To reproduce the online point experiments described in Figure 7,
first train a model for each benchmark as before, if not already done.

To measure each benchmark's unoptimized runtime, run each with Willump disabled:

    python3 tests/benchmark_scripts/toxic_point.py -d
    python3 tests/benchmark_scripts/product_point.py -d
    python3 tests/benchmark_scripts/music_point.py -d
    python3 tests/benchmark_scripts/music_remote_point_setup.py -d -r REDIS_SERVER_ADDRESS && python3 tests/benchmark_scripts/music_remote_point_eval.py -d -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/credit_point.py -d
    python3 tests/benchmark_scripts/price_point.py -d
    python3 tests/benchmark_scripts/instant_point.py -d
    
To measure the effects of Willump's compilation optimizations on runtime for compilable benchmarks, run:

    python3 tests/benchmark_scripts/toxic_point.py
    python3 tests/benchmark_scripts/product_point.py
    python3 tests/benchmark_scripts/music_point.py
    python3 tests/benchmark_scripts/credit_point.py
    python3 tests/benchmark_scripts/price_point.py
    
To measure the effects of Willump's cascades optimization on runtime for classification benchmarks, run:

    python3 tests/benchmark_scripts/toxic_point.py -c
    python3 tests/benchmark_scripts/product_point.py -c
    python3 tests/benchmark_scripts/music_point.py -c
    python3 tests/benchmark_scripts/music_remote_point_setup.py -c -r REDIS_SERVER_ADDRESS && python3 tests/benchmark_scripts/music_remote_point_eval.py -c -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/instant_point.py -c

Each benchmark reports its median and tail latencies.
For each benchmark, latency trends should resemble those in Figure 7.

### Figure 8

To reproduce the top-k experiments described in Figure 8,
first train a top-k model for each benchmark:

    python3 tests/benchmark_scripts/toxic_train.py -k 100
    python3 tests/benchmark_scripts/product_train.py -k 100
    python3 tests/benchmark_scripts/music_train.py -k 100
    python3 tests/benchmark_scripts/credit_train.py -k 100
    python3 tests/benchmark_scripts/price_train.py -k 100
    python3 tests/benchmark_scripts/instant_train.py -k 100

To measure each benchmark's unoptimized runtime, run each with Willump disabled:

    python3 tests/benchmark_scripts/toxic_topk.py -k 100 -d
    python3 tests/benchmark_scripts/product_topk.py -k 100 -d
    python3 tests/benchmark_scripts/music_topk.py -k 100 -d
    python3 tests/benchmark_scripts/music_remote_topk.py -k 100 -d -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/credit_remote_topk.py -k 100 -d -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/price_topk.py -k 100 -d
    python3 tests/benchmark_scripts/instant_topk.py -k 100 -d
    
To measure the effects of Willump's compilation optimizations on runtime for compilable benchmarks, run:

    python3 tests/benchmark_scripts/toxic_topk.py -k 100
    python3 tests/benchmark_scripts/product_topk.py -k 100
    python3 tests/benchmark_scripts/music_topk.py -k 100
    python3 tests/benchmark_scripts/price_topk.py -k 100
    
To measure the effects of Willump's top-k approximation optimization on runtime, run:

    python3 tests/benchmark_scripts/toxic_topk.py -k 100 -c
    python3 tests/benchmark_scripts/product_topk.py -k 100 -c
    python3 tests/benchmark_scripts/music_topk.py -k 100 -c
    python3 tests/benchmark_scripts/music_remote_topk.py -k 100 -c -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/credit_remote_topk.py -k 100 -c -r REDIS_SERVER_ADDRESS
    python3 tests/benchmark_scripts/price_topk.py -k 100 -c
    python3 tests/benchmark_scripts/instant_topk.py -k 100 -c

Each benchmark reports its throughput.
For each benchmark, throughput trends should resemble those in Figure 8.