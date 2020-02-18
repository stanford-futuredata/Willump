#!/usr/bin/env bash

echo "Training Models and Cascades"

python3 tests/benchmark_scripts/toxic_train.py | awk NF
python3 tests/benchmark_scripts/product_train.py | awk NF
python3 tests/benchmark_scripts/music_train.py | awk NF

echo "Running Batch Benchmarks -- Figure 6"

echo "Toxic Batch Unoptimized:"
python3 tests/benchmark_scripts/toxic_batch.py -d
echo "Toxic Batch Compiled:"
python3 tests/benchmark_scripts/toxic_batch.py | awk NF
echo "Toxic Batch Cascaded:"
python3 tests/benchmark_scripts/toxic_batch.py -c | awk NF

echo "Product Batch Unoptimized:"
python3 tests/benchmark_scripts/product_batch.py -d
echo "Product Batch Compiled:"
python3 tests/benchmark_scripts/product_batch.py | awk NF
echo "Product Batch Cascaded:"
python3 tests/benchmark_scripts/product_batch.py -c | awk NF

echo "Music Batch Unoptimized:"
python3 tests/benchmark_scripts/music_batch.py -d
echo "Music Batch Compiled:"
python3 tests/benchmark_scripts/music_batch.py | awk NF
echo "Music Batch Cascaded:"
python3 tests/benchmark_scripts/music_batch.py -c | awk NF

echo "Running Point Benchmarks -- Figure 7"

echo "Toxic Point Unoptimized:"
python3 tests/benchmark_scripts/toxic_point.py -d
echo "Toxic Point Compiled:"
python3 tests/benchmark_scripts/toxic_point.py | awk NF
echo "Toxic Point Cascaded:"
python3 tests/benchmark_scripts/toxic_point.py -c | awk NF

echo "Product Point Unoptimized:"
python3 tests/benchmark_scripts/product_point.py -d
echo "Product Point Compiled:"
python3 tests/benchmark_scripts/product_point.py | awk NF
echo "Product Point Cascaded:"
python3 tests/benchmark_scripts/product_point.py -c | awk NF

echo "Music Point Unoptimized:"
python3 tests/benchmark_scripts/music_point.py -d
echo "Music Point Compiled:"
python3 tests/benchmark_scripts/music_point.py | awk NF
echo "Music Point Cascaded:"
python3 tests/benchmark_scripts/music_point.py -c | awk NF

echo "Training TopK Models and Approximations"

python3 tests/benchmark_scripts/toxic_train.py -k 100 | awk NF
python3 tests/benchmark_scripts/product_train.py -k 100 | awk NF
python3 tests/benchmark_scripts/music_train.py -k 100 | awk NF

echo "Running Top-K Benchmarks -- Figure 8"

echo "Toxic Top-K Unoptimized:"
python3 tests/benchmark_scripts/toxic_topk.py -d -k 100
echo "Toxic Top-K Compiled:"
python3 tests/benchmark_scripts/toxic_topk.py -k 100 | awk NF
echo "Toxic Top-K Cascaded:"
python3 tests/benchmark_scripts/toxic_topk.py -c -k 100 | awk NF

echo "Product Top-K Unoptimized:"
python3 tests/benchmark_scripts/product_topk.py -d -k 100
echo "Product Top-K Compiled:"
python3 tests/benchmark_scripts/product_topk.py -k 100 | awk NF
echo "Product Top-K Cascaded:"
python3 tests/benchmark_scripts/product_topk.py -c -k 100 | awk NF

echo "Music Top-K Unoptimized:"
python3 tests/benchmark_scripts/music_topk.py -d -k 100
echo "Music Top-K Compiled:"
python3 tests/benchmark_scripts/music_topk.py -k 100 | awk NF
echo "Music Top-K Cascaded:"
python3 tests/benchmark_scripts/music_topk.py -c -k 100 | awk NF