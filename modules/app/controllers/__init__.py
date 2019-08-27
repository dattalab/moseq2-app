''' all controllers for various collections of database '''
import os
import glob

# import all the routes in all the files inside this 'controllers' directory
__all__ = [os.path.basename(f)[:-3]
           for f in glob.glob(os.path.dirname(__file__) + '/*.py')]
