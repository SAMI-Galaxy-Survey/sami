import pytest

from sami.qc import fluxcal

def test_get_sdss_data():
    result = fluxcal.get_sdss_stellar_mags(["a"], [(15.5, 0.5)], automatic=True)

    print(result)

    assert result == ("name,objID,ra,dec,type,psfMag_u,psfMagErr_u,psfMag_g,psfMagErr_g,psfMag_r,psfMagErr_r,psfMag_i,psfMagErr_i,psfMag_z,psfMagErr_z\n" +
    "a,1237663784741045437,15.5022412174027,0.493596063748454,GALAXY,23.44828,0.3526277,23.84825,0.3183443,23.13127,0.1956578,22.37959,0.161695,21.76724,0.8159173")
