import numpy as np
import h5py

def get_mom_mat(matl_path: str):
    """
    Obtains the complex dipole moment matrix elements and shortens the list based on the number of particpating bands for excitonic transitions.
    
    Args:
        BSE_path (str): Path to BSE folder

    Returns:
        mom_mat (numpy.ndarray): Array of dipole moment matrix elements in the format of (nband_v, nband_c, kpt_coord, ispin, directional_element)
    """
    with open(matl_path + "/4-BSE/WAVEDER", "rb") as fp:
        # Header Data
        ## Read n-bands_tot, n-bands_contrib, n-kpts, i-spin
        prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        [nbands, nbands_contrib, nibzkpts, ispin] = np.fromfile(fp, dtype=np.int32, count=4)
        suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        if(abs(suffix)-abs(prefix)):
            print("Read incorrect number of bytes")
            print("Expected: ", prefix, "Read: 32")

        ## nodes in dielectric function - part of header, not used
        prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        np.fromfile(fp, dtype=np.float64, count=1)
        suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        if(abs(suffix)-abs(prefix)):
            print("Read incorrect number of bytes")
            print("Expected: ", prefix, "Read: 8")

        ## WPLASMON - part of header, not used
        prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        np.fromfile(fp, dtype=np.float64, count=9)
        suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        if(abs(suffix)-abs(prefix)):
            print("Read incorrect number of bytes")
            print("Expected: ", prefix, "Read: 72")

        # Body Data
        ## Matrix Elements (mom_mat or rijks)
        mom_mat = []
        prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        mom_mat.append(np.fromfile(fp, dtype=np.complex64, count=3*nbands*nbands_contrib*nibzkpts*ispin))
        suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        if(abs(suffix)-abs(prefix)):
            print("Read incorrect number of bytes")
            print("Expected: ", prefix, "Read: ", 3*nbands*nbands_contrib*ispin*nibzkpts*8)

        ## Reshape the matrix according to n-bands_tot, n-bands_contrib, n-kpts, i-spin
        mom_mat = np.array(mom_mat)
        mom_mat = mom_mat.reshape(3, ispin, nibzkpts, nbands_contrib, nbands).T

    with h5py.File(matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
        val_num_max = f_h5["results"]["linear_response"]["bse_index"].shape[3]
        cond_num_max = val_num_max + f_h5["results"]["linear_response"]["bse_index"].shape[2]

    return mom_mat[0:val_num_max, 0:cond_num_max, :, :, :]