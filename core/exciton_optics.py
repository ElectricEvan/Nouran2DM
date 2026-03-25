import h5py
import numpy as np
import polars as pl
from core.exceptions import FBZKPTsMismatchError


def exciton_optics(matl_path: str, df_fatbands: pl.DataFrame, mom_mat: np.ndarray, light_polar: list = 0):
    '''
    Calculates the BSE oscilator strength of an exciton 

    Args:
        BSE_path (str): Path to BSE folder
        GW_path (str): Path to GW folder
        df_fatbands (polars.DataFrame): Polars DataFrame of all excitonic transitions for one or multiple excitons
        mom_mat (numpy.ndarray): Numpy array of dipole moment matrix elements in the format of (nband_v, nband_c, kpt_coord, ispin, directional_element)
    
    Returns:
        bse_strengths (list): A list of calculated BSE oscillator strengths
    '''
    # Get relevant data
    with h5py.File(matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
        fbzkpts = f_h5["results"]["electron_eigenvalues"]["kpoint_coords"][:]
        n_exc_trans = f_h5["results"]["linear_response"]["bse_fatbands"].shape[1]
        n_exc = int(len(df_fatbands)/n_exc_trans)
        # exc_bandgaps = f_h5["results"]["linear_response"]["opticaltransitions"][0:n_exc, 0]
        
    # Get KPT symmetry mapping, symmetry ops, and relevant data
    with h5py.File(matl_path + "/3-GW/vaspout.h5", "r") as f_h5:
        # Relevant data
        lat_vect = np.array(f_h5["results"]["positions"]["lattice_vectors"])

        # Symmetry stuff
        fbzkpts_GW = np.array(f_h5["results"]["electron_eigenvalues"]["kpoint_coords_full"][:])
        kpt_mapping = np.array(f_h5["results"]["electron_eigenvalues"]["kpoints_symmetry_mapping"]).reshape(-1, 1)
        kpt_symops = np.array(f_h5["results"]["electron_eigenvalues"]["kpoints_symmetry_symop"])

        # Fix the indexing of the FBZKPTS in GW
        fbzkpts_GW_sort_correction = np.argsort(np.lexsort((fbzkpts[:, 0], fbzkpts[:, 1], fbzkpts[:, 2])))
        fbzkpts_GW_sort_ind = np.lexsort((fbzkpts_GW[:, 0], fbzkpts_GW[:, 1], fbzkpts_GW[:, 2]))
        fbzkpts_GW = fbzkpts_GW[fbzkpts_GW_sort_ind][fbzkpts_GW_sort_correction]
        if np.array_equal(fbzkpts, fbzkpts_GW):
            kpt_mapping = kpt_mapping[fbzkpts_GW_sort_ind][fbzkpts_GW_sort_correction]
            kpt_symops = kpt_symops[fbzkpts_GW_sort_ind][fbzkpts_GW_sort_correction]
            del fbzkpts_GW, fbzkpts_GW_sort_correction
            kpt_mapping = kpt_mapping[:, -1]
        else:
            raise FBZKPTsMismatchError(f"The FBZKPTs from BSE dataset is different from the GW dataset")

    # Merge real and imaginary parts of X_BSE and cleanup df for speed
    df_fatbands = df_fatbands.with_columns(pl.Series("X_BSE", df_fatbands[:, -2].to_numpy() + df_fatbands[:, -1].to_numpy()*(1j))).drop(["Re(X_BSE)", "Im(X_BSE)"])
    df_fatbands = df_fatbands.drop(["E_v", "E_c", "Abs(X_BSE)/W_k"])

    # Replicate the control data
    ## Tabulate rxyz_col and S_col
    rxyz_col = []
    Sxyz_col = []
    fbzkpts = pl.DataFrame(fbzkpts)
    j = 0
    for i in range(len(df_fatbands[0:n_exc_trans])):
        if not np.array_equal(df_fatbands[i, 0:3], fbzkpts[j]):
            j += 1
        Sxyz = lat_vect.T @ kpt_symops[j] @ np.linalg.inv(lat_vect.T)
        rxyz = Sxyz @ mom_mat[df_fatbands[i, 3]-1, df_fatbands[i, 4]-1, kpt_mapping[j]-1][0]
        Sxyz_col.append(Sxyz)
        rxyz_col.append(rxyz)

    rxyz_col *= n_exc
    rxyz_col = np.array(rxyz_col)
    df_fatbands = df_fatbands.with_columns(pl.Series("rx", rxyz_col[:, 0]))
    df_fatbands = df_fatbands.with_columns(pl.Series("ry", rxyz_col[:, 1]))
    df_fatbands = df_fatbands.with_columns(pl.Series("rz", rxyz_col[:, 2]))
    df_fatbands = df_fatbands.with_columns(pl.col(["Kx", "Ky", "Kz"]).round(6)) # Rounding kpoint coords to filter easier


    ## Calculate the BSE strengths
    bse_strengths_test = []
    r_polar_strength = []
    l_polar_strength = []
    non_polar_strength = []
    non_polar_strength_norm = []
    planar_strength = []

    # df_fatbands = df_fatbands.filter((pl.col("Kx") == 0.333333) & (pl.col("Ky") == 0.333333) & (pl.col("Kz") == 0))
    v_min = df_fatbands["nbands_v"].min()
    v_max = df_fatbands["nbands_v"].max()
    c_min = df_fatbands["nbands_c"].min()
    c_max = df_fatbands["nbands_c"].max()

    for i in range(n_exc):
        exc_dipole_vector = []
        for a in range(rxyz_col.shape[1]): # a = x, y, z
            exc_dipole_vector_a = 0
            for v in range(v_min, v_max+1):
                for c in range(c_min, c_max+1):
                    # Sum all c, v for all kpoints by filtering for c and v in the df
                    df_fatbands_bands_filtered = df_fatbands[i*n_exc_trans:(i+1)*n_exc_trans].filter((pl.col("nbands_v") == v) & (pl.col("nbands_c") == c))    # Sum Ks
                    df_fatbands_bands_filtered = df_fatbands_bands_filtered[:, -4:].to_numpy()
                    exc_dipole_vector_a += np.sum(df_fatbands_bands_filtered[:, 1+a] * (df_fatbands_bands_filtered[:, 0])) # matrix element * bse vector

            exc_dipole_vector.append(exc_dipole_vector_a)
    

        # bse_strengths_test.append(float(sum([abs(comp**2) for comp in exc_dipole_vector])))
        # r_light_polar = np.array([1/2**0.5, 1j*1/2**0.5, 0])
        # l_light_polar = np.array([1/2**0.5, -1j*1/2**0.5, 0])
        # non_polar = np.array([1, 0, 0])
        # non_polar_norm = np.array([1/3, 1/3, 1/3])
        # planar = np.array([1/2**0.5, 1/2**0.5, 0])
        # r_polar_strength.append(float(abs(r_light_polar @ np.array(exc_dipole_vector)**2)))
        # l_polar_strength.append(float(abs(l_light_polar @ np.array(exc_dipole_vector)**2)))
        # non_polar_strength.append(float(abs(non_polar @ np.array(exc_dipole_vector)**2)))
        # non_polar_strength_norm.append(float(abs(non_polar_norm @ np.array(exc_dipole_vector)**2)))
        # planar_strength.append((float(abs(planar @ np.array(exc_dipole_vector)**2))))

    print(f"{r_polar_strength=}")
    print(f"{l_polar_strength=}")
    print(f"{non_polar_strength=}")
    print(f"{non_polar_strength_norm=}")
    print(f"{planar_strength=}")

    return bse_strengths_test
    

def verify(matl_path: str, bse_strengths: list, verbose: bool = False):
    '''
    Docstring for verify
    
            verify (bool): If True, reads the BSE oscillator strength from vaspout.h5 as a control to verify the this library's results to it, and prints out the \
            verification results. Defaults to Falseif verify True, a list containing boolean expressions for if \
        the calculated BSE oscillator strengths match the control. None if False.
    '''
    # Get relevant data
    with h5py.File(matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
        n_exc = len(bse_strengths)
        bse_strengths_ctrl = f_h5["results"]["linear_response"]["opticaltransitions"][0:n_exc, 1]

    verify_ls = np.isclose(bse_strengths_ctrl, bse_strengths, rtol=1E-13, atol=1E-13).tolist()

    if verbose:
        print("\n--- VERIFICATION RESULTS: ---")
        print(f"CTRL: {bse_strengths_ctrl.tolist()}")
        print(f"CALCULATED: {bse_strengths}\n")
        print(verify_ls)

    return verify_ls

