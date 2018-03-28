========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/sami_package/badge/?style=flat
    :target: https://readthedocs.org/projects/sami_package
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/astrogreen/sami_package.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/astrogreen/sami_package

.. |version| image:: https://img.shields.io/pypi/v/sami.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/sami

.. |commits-since| image:: https://img.shields.io/github/commits-since/astrogreen/sami_package/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/astrogreen/sami_package/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/sami.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/sami

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/sami.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/sami

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/sami.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/sami


.. end-badges

The SAMI python package contains functionality for running the data reduction process described in Sharp et al. (2014). See also Allen et al. (2014) for an assessment of the quality of the finished products.

There is currently no automated installation. Instead you should place the source code in your `PYTHONPATH` and make sure you have the following packages installed:

* numpy
* scipy
* matplotlib
* astropy
* bottleneck (optional)

The initial data reduction steps also make use of 2dfdr, which is developed by the Australian Astronomical Observatory and [available here](http://www.aao.gov.au/science/software/2dfdr).


Installation
============

::

    pip install sami

Documentation
=============

https://sami_package.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
