"""
This package contains the data structures for Willump's graphs.  A Willump graph is a representation
of a pipeline as a DAG of transforms.  Each node is a transform, each edge is a materialized
dataframe.  Sources are inputs and the sink is the output.
"""