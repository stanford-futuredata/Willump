# Willump

**Willump is unstable and under active development!  Please report any bugs or unusual behavior to [willump-group@cs.stanford.edu](mailto:willump-group@cs.stanford.edu).**

Willump is an optimizer for machine learning inference.  It speeds up ML inference pipelines written
in Python using Numpy and Pandas.  To speed up a function using Willump, simply
wrap it in the Willump decorator: 

    from willump.evaluation.willump_executor import willump_execute
    
    
    @willump_execute()
    def make_me_faster(...):
    
To install Willump, first install the llvm-st branch of our Weld fork, weld-willump.
Its repository and installation instructions are available 
[here](https://github.com/stanford-futuredata/weld-willump/tree/llvm-st).

Next, install Python, Mypy, NumPy, scikit-learn, and Pandas.  Willump requires Python
version 3.6 or later.

Next, define the WILLUMP_HOME environment variable to point
to the Willump root directory (this one) and include the Willump root directory
on your PYTHONPATH.  Willump should now work!

To confirm Willump works, run the Willump unit tests:

    python3 -m unittest discover -s tests -p *.py

To run our Willump benchmarks, first train the benchmark models:

    python3 tests/test_scripts/wsdm_music_recommendation_example_train.py
    python3 tests/test_scripts/lazada_product_challenge_example_train.py
    
Then run the benchmarks themselves using the trained models:

    python3 tests/test_scripts/wsdm_music_recommendation_example_batch.py
    python3 tests/test_scripts/lazada_product_challenge_example_batch.py
    
Both benchmarks should run dramatically faster than they would without Willump.
