# Willump

**Willump is unstable and under active development!  Please report any bugs or unusual behavior to [willump-group@cs.stanford.edu](mailto:willump-group@cs.stanford.edu).**

Willump is an optimizer for machine learning inference.  It speeds up ML inference pipelines
written as Python functions whose performance is bottlenecked by feature computation.
For a full description of Willump, see [our paper](https://arxiv.org/pdf/1906.01974.pdf).

## Installation

Willump requires Python version 3.6 or later.
These instructions were tested on a clean installation of Ubuntu 18.04 with Python 3.6.8 installed.

First, install dependency packages:

    sudo apt update
    sudo apt install build-essential curl python3-pip
    pip3 install numpy scipy sklearn pandas astor setuptools tqdm pandas redis lightgbm==2.2.2 tensorflow==1.12.0 keras==2.2.4
    
Then install the llvm-st branch of our Weld fork, weld-willump.
Its repository and installation instructions are available 
[here](https://github.com/stanford-futuredata/weld-willump/tree/llvm-st).

Copy the weld-willump libraries to /usr/lib so clang can find them:

    sudo cp $WELD_HOME/target/release/libweld.so /usr/lib/libweld.so
    
Install the weld-willump Python libraries:

    cd $WELD_HOME/python/pyweld
    sudo -E python3 setup.py install

Finally, clone Willump and set the WILLUMP_HOME and PYTHONPATH environment variables
to point at and include its root.

    git clone https://github.com/stanford-futuredata/Willump.git
    cd Willump
    export WILLUMP_HOME=`pwd`
    export PYTHONPATH=$PYTHONPATH:`pwd`

To confirm Willump works, run the Willump unit tests:

    python3 -m unittest discover -s tests -p *.py

For information on reproducing the experiments run in the Willump paper, see our
[benchmarks guide](https://github.com/stanford-futuredata/Willump/blob/master/BENCHMARKS.md)