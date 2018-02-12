#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name='sami',
      version='1.0',
      description='Data reduction pipeline for the SAMI instrument',
      author="The SAMI Galaxy Survey Team",
      author_email='Please refer to the ``maintainer_email``',
      maintainer="Francesco D'Eugenio",
      maintainer_email='francesco.deugenio@anu.edu.au',
      url='https://bitbucket.org/james_t_allen/sami-package',
      py_modules=['manager', 'samifitting', 'update_csv'],
      packages=['utils', 'dr', 'qc'],
      ext_modules=[Extension('utils.circ', ['utils/cCirc.cc']),
                   Extension('general.covar', ['general/cCovar.cc'])]
     )
