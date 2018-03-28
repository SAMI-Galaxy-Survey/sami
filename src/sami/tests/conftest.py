# content of conftest.py

# Word for word from:
#
#       http://pytest.org/latest/example/simple.html#incremental-testing-test-steps

import os

from six import PY2
from six.moves import urllib
import shutil
import subprocess

import pytest

TEST_DIR = os.path.join(os.path.split(__file__)[0], "test_data")

def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" %previousfailed.name)

@pytest.fixture(scope='session')
def raw_test_data():
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    if not os.path.exists(os.path.join(TEST_DIR, "sami_raw_test_data")):
        data_uri = "http://db.sami-survey.org/sami_raw_test_data.tar.gz"
        filename = os.path.join(TEST_DIR, "sami_raw_test_data.tar.gz")
        urllib.request.urlretrieve(data_uri, filename)
        if PY2:
            subprocess.check_output(["tar", "-xzvf", "sami_raw_test_data.tar.gz"], cwd=TEST_DIR)
        else:
            shutil.unpack_archive(filename, TEST_DIR)
    else:
        pass

    return os.path.join(TEST_DIR, "sami_raw_test_data")
