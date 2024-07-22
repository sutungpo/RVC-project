import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = 'gui_package',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gui_package",  ["gui_package.py"]),]
)
# setup(
#     name = 'Img_dispose',
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("Img_dispose",  ["Img_dispose.py"]),]
# )
