#define a bootstrapping function to import the python module (necessary on the heroku server). Mostly boilerplate from pibind documentation

def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'FFNN_pymodule.cpython-38-x86_64-linux-gnu.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()