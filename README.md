# Willump
Willump is a system for fast featurization.  Willump speeds up featurization functions written in
Python.  To speed up a function using Willump, simply wrap it in the Willump decorator:

    import willump.evaluation.willump_executor
    
    
    @willump.evaluation.willump_executor.willump_execute
    def make_me_faster(...):
    
Willump currently works on very few types of functions, but we're improving it every day!

For Willump to work, you must define the WILLUMP_HOME environment variable to point to the Willump
root directory (this one).  You must also include the Willump root directory on your PYTHONPATH.