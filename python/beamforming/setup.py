from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

ext_modules=[
    Extension("cbf_mac_mp",
              ["cbf_mac_mp.pyx"],
              libraries=["m"],
              #extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_compile_args = ["-O3", "-ffast-math", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup( 
  name = "cbf_mac_mp",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)

ext_modules_alt=[
    Extension("cbf_mac_alt_mp",
              ["cbf_mac_alt_mp.pyx"],
              libraries=["m"],
              #extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_compile_args = ["-O3", "-ffast-math", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup( 
  name = "cbf_mac_alt_mp",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules_alt
)


setup(ext_modules = cythonize("cbf_mac.pyx"))

setup(ext_modules = cythonize("cbf_mac_alt.pyx"))
