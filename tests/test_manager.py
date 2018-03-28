from __future__ import print_function

import pytest

from tempfile import mkdtemp
from glob import glob
import shutil
import fnmatch
import os
import os.path

import sami

TEST_DIR = os.path.join(os.path.split(__file__)[0], "test_data")

# Note: if the test data is changed, then these lists must be updated
# (too hard to automate!)
tlm_files = ("22apr10074", "22apr20074")
flat_files = tlm_files
arc_files = ("22apr10075", "22apr20075")
obj_files = ("22apr10078", "22apr20078", "22apr10079", "22apr20079")

def find_files(path, pattern):
    """From: 

    http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python

    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(filename)
    return matches

@pytest.fixture(scope='module')
def reduction_dir(request):
    tmpdir = mkdtemp(prefix="sami_test")
    print(tmpdir)
    def fin():
        #shutil.rmtree(tmpdir)
        pass
    request.addfinalizer(fin)

    return tmpdir

@pytest.mark.incremental
class TestSAMIManagerReduction:

    @pytest.fixture
    def sami_manager(self, reduction_dir):
        mngr = sami.manager.Manager(reduction_dir + "/test/", fast=True, debug=True)
        return mngr

    def test_pytest_not_capturing_fds(self, pytestconfig):
        # Note: pytest must be run in sys capture mode, instead of file descriptor capture mode
        # otherwise calls to "aaorun" seem to fail. This next test ensures that is the case.
        print("If this test fails, then you must run pytest with the option '--capture=sys'.")
        assert pytestconfig.getoption("capture") == "sys"

    def test_tests(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager
        assert isinstance(mngr, sami.manager.Manager)
        print(reduction_dir)
        assert isinstance(reduction_dir, str)
        # assert os.path.exists(reduction_dir + "/test")

    def test_import_data(self, sami_manager, raw_test_data):
        mngr = sami_manager  # type: sami.Manager
        mngr.import_dir(raw_test_data)
        print(len(mngr.file_list))
        assert len(mngr.file_list) == 8

    def test_make_tlm(self, sami_manager, raw_test_data, reduction_dir, capfd):
        mngr = sami_manager
        mngr.make_tlm()

        # Check that files actually generated
        for base in tlm_files:
            assert base + "tlm.fits" in find_files(reduction_dir + "/test/", base + "*")


    def test_reduce_arc(self, sami_manager, raw_test_data, reduction_dir, capfd):
        mngr = sami_manager
        mngr.reduce_arc()
        # Check that files actually generated
        for base in arc_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")

    def test_reduce_fflat(self, sami_manager, raw_test_data, reduction_dir, capfd):
        mngr = sami_manager
        mngr.reduce_fflat()
        for base in flat_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")


    def test_reduce_object(self, sami_manager, raw_test_data, reduction_dir, capfd):
        mngr = sami_manager
        mngr.reduce_object()
        for base in obj_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")
