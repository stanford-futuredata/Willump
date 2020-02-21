FROM python:3.6-slim-stretch

RUN mkdir -p /python-deps/Willump

RUN apt-get update
RUN apt-get install -y build-essential curl
RUN apt-get update

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

RUN apt-get install -y zlib1g-dev wget software-properties-common
RUN wget https://apt.llvm.org/llvm-snapshot.gpg.key -O bob.key
RUN apt-key add bob.key
RUN apt-add-repository "deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-6.0 main"
RUN apt-get update
RUN apt-get install -y llvm-6.0-dev clang-6.0
RUN ln -s /usr/bin/llvm-config-6.0 /usr/local/bin/llvm-config
RUN ln -s /usr/bin/clang++-6.0 /usr/local/bin/clang++

RUN apt-get install -y gcc git
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/stanford-futuredata/weld-willump.git
WORKDIR "/weld-willump"
RUN git checkout llvm-st
RUN cargo build --release

ENV WELD_HOME=/weld-willump
RUN pip install setuptools
WORKDIR "/weld-willump/python/pyweld"
RUN python setup.py install
WORKDIR "/"
RUN cp weld-willump/target/release/libweld.so /usr/lib/libweld.so

ADD $WILLUMP_HOME/tests /python-deps/Willump/tests
ADD $WILLUMP_HOME/willump /python-deps/Willump/willump
ADD $WILLUMP_HOME/cppextensions /python-deps/Willump/cppextensions
ADD $WILLUMP_HOME/requirements.txt /python-deps/Willump
ADD $WILLUMP_HOME/setup.py /python-deps/Willump
ADD $WILLUMP_HOME/simple_benchmarks.sh /python-deps/Willump

ENV WILLUMP_HOME=/python-deps/Willump

WORKDIR "/python-deps/Willump"
RUN pip install -r requirements.txt
RUN python setup.py install --user
