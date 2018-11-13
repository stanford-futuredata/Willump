from distutils.core import setup, Extension


call_weld_llvm_module = Extension("weld_llvm_caller",
                                        include_dirs=["/usr/include", "/usr/local/include"],
                                        libraries=['weld'],
                                        library_dirs=['/usr/local/lib', "/usr/lib"],
                                        sources=['cppextensions/weld_llvm_caller.cpp'])
setup(name='WillumpExtension',
          ext_modules=[call_weld_llvm_module])