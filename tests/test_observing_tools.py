import pytest

import os

import sami

TEST_DIR = os.path.dirname(__file__)


def test_offset_hexa(capsys):

    sami.utils.other.offset_hexa(os.path.join(TEST_DIR, "example_allocated_csv.csv"))

    out, err = capsys.readouterr()

    assert out == """----------------------------------------------------------------------
Move the telescope 219.3 arcsec E and 49.9 arcsec N
The star will move from the central hole
    to guide bundle 2 (nG1 on plate)
    (alternately, set the APOFF to X:49.9, Y:219.3)
Move the telescope 710.8 arcsec W and 547.0 arcsec S
The star will move from the central hole
    to hexabundle 7 (nP11 on plate)
----------------------------------------------------------------------
"""

