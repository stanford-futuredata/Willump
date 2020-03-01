# Willump

**Willump is unstable and under active development!  Please report any bugs or unusual behavior.**

Willump is an optimizer for machine learning inference.  It speeds up ML inference pipelines
written as Python functions whose performance is bottlenecked by feature computation.
For a high-level summary of Willump, please see [this blog post](https://dawn.cs.stanford.edu/2020/02/29/willump/).
For a full description, please see [our paper](http://petereliaskraft.net/res/willump.pdf),
published at [MLSys 2020](https://mlsys.org).

## Installation

Willump requires Python version 3.6 or later.
These instructions were tested on a clean installation of Ubuntu 18.04 with Python 3.6.8.

First, install dependency packages:

    sudo apt update
    sudo apt install build-essential curl python3-pip
    pip3 install setuptools
    
Then install the llvm-st branch of our Weld fork, weld-willump.
Its repository and installation instructions are available 
[here](https://github.com/stanford-futuredata/weld-willump/tree/llvm-st).

Copy the weld-willump libraries to /usr/lib so clang can find them:

    sudo cp $WELD_HOME/target/release/libweld.so /usr/lib/libweld.so
    
Install the weld-willump Python libraries:

    cd $WELD_HOME/python/pyweld
    sudo -E python3 setup.py install

Finally, clone Willump, set the WILLUMP_HOME environment variable to point at the package root, and include
the package root in your PYTHONPATH:

    git clone https://github.com/stanford-futuredata/Willump.git
    cd Willump
    pip3 install -r requirements.txt
    export WILLUMP_HOME=`pwd`
    python3 setup.py install --user

To confirm Willump works, run the Willump unit tests:

    python3 setup.py test

For information on reproducing the experiments run in the Willump paper, please see our
[benchmarks guide](https://github.com/stanford-futuredata/Willump/blob/master/BENCHMARKS.md).

## Docker

You can also experiment with Willump using Docker with the Dockerfile we provide.  To clone our repository
and build our container, run:

    git clone https://github.com/stanford-futuredata/Willump.git
    cd Willump
    export WILLUMP_HOME=`pwd`
    docker build -t willump .

Then, to verify the container built successfully, run our unit tests:

    docker run -t willump python setup.py test
