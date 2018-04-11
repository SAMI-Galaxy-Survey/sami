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
bias_files = ("22apr10035", "22apr10036", "22apr10037",
              "22apr20035", "22apr20036", "22apr20037")
dark_files = ("22apr10001", "22apr10002", "22apr10003",
              "22apr20001", "22apr20002", "22apr20003")
lflat_files = ("14apr10027", "22apr10088",
               "14apr20027", "22apr20088")
tlm_files = ("22apr10074", "22apr20074")
flat_files = tlm_files
arc_files = ("22apr10075", "22apr20075")
obj_files = ("22apr10078", "22apr20078", "22apr10079", "22apr20079")

all_files = set(bias_files + dark_files + lflat_files + tlm_files + flat_files + arc_files + obj_files)

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
        print(len(all_files))
        assert len(mngr.file_list) == len(all_files)

    def test_reduce_bias(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.reduce_bias()

        # Check that files actually generated
        for base in bias_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/reduced/bias", base + "*")

    def test_combine_bias(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.combine_bias()

        # Check that files actually generated
        assert "BIAScombined.fits" in find_files(reduction_dir + "/test/reduced/bias/ccd_1", "*.fits")
        assert "BIAScombined.fits" in find_files(reduction_dir + "/test/reduced/bias/ccd_2", "*.fits")

    def test_reduce_dark(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.reduce_dark()

        # Check that files actually generated
        for base in dark_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/reduced/dark", base + "*")

    def test_combine_dark(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.combine_dark()

        # Check that files actually generated
        assert "DARKcombined1800.fits" in find_files(reduction_dir + "/test/reduced/dark/ccd_1", "*.fits")
        assert "DARKcombined1800.fits" in find_files(reduction_dir + "/test/reduced/dark/ccd_2", "*.fits")

    def test_reduce_lflat(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.reduce_lflat()

        # Check that files actually generated
        for base in lflat_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/reduced/lflat", base + "*")

    def test_combine_lflat(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager  # type: sami.Manager
        mngr.combine_lflat()

        # Check that files actually generated
        assert "LFLATcombined.fits" in find_files(reduction_dir + "/test/reduced/lflat/ccd_1", "*.fits")
        assert "LFLATcombined.fits" in find_files(reduction_dir + "/test/reduced/lflat/ccd_2", "*.fits")

    def test_make_tlm(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager
        mngr.make_tlm()

        # Check that files actually generated
        for base in tlm_files:
            assert base + "tlm.fits" in find_files(reduction_dir + "/test/", base + "*")


    def test_reduce_arc(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager
        mngr.reduce_arc()
        # Check that files actually generated
        for base in arc_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")

    def test_reduce_fflat(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager
        mngr.reduce_fflat()
        for base in flat_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")


    def test_reduce_object(self, sami_manager, raw_test_data, reduction_dir):
        mngr = sami_manager
        mngr.reduce_object()
        for base in obj_files:
            assert base + "red.fits" in find_files(reduction_dir + "/test/", base + "*")
