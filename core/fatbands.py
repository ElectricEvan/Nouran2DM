import numpy as np
import polars as pl
import h5py
from core.exceptions import ExcitonRangeError

def valid_transitions(matl_path: str, n_exc: int=0):
    """
    Gets all excitonic transitions from BSE vaspout.h5 that are below Omega_max (max energy difference for excitation pairs). These transistions include all the various
    points (Kx, Ky, Kz), energy levels the valence and conduction bands (E_v, E_c) within Omega_max difference, the relative strength of the exciton's component at that 
    kpoint (Abs(X_BSE)/W_k), the valence and conduction band orbital numbers (nbands_v, nbands_c), and the complex amplitude of the e-h bound state (Re(X_BSE), Im(X_BSE)). 
    Although, this information is provided by BSEFATBAND, it loses precision from rounding values and may not always be included.

    Args:
        BSE_path (str): Path to BSE folder
        verify (bool): If True, opens BSEFATBAND as a control to verify if this library's results matches with it, and prints a sample of results from random indices. \
            Defaults to False. 
        n_exc (int): The number of excitonic bangaps starting from the lowest value. Defaults to 1

    Returns:
        (df_trans, verfify_ls) (tuple): Containing a DataFrame of all excitonic transitions' kpoint coordinates, valence/conduction band index, and complex BSE amplitude \
            obtained from vaspout.h5 and, if verify is True, a numpy.ndarray containing tuples of every column in BSEFATBANDS and boolean expressions for if that column \
                replicates the control file correctly. None if False
    """
    # Open all relevant files
    with h5py.File(matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
        kpoint_weight = f_h5["results"]["electron_eigenvalues"]["kpoints_symmetry_weight"][0]
        exc_count = f_h5["results"]["linear_response"]["bse_fatbands"].shape[0]
        n_exc_trans = f_h5["results"]["linear_response"]["bse_fatbands"].shape[1]

        # Check if n_exc is within a valid range
        if n_exc > exc_count or n_exc < 0:
            raise ExcitonRangeError(f"{n_exc} n_exc is out of range for VASP calculations which only contains {exc_count} excitons.")

        fatbands = f_h5["results"]["linear_response"]["bse_fatbands"][0:n_exc, :, :].reshape(-1,2)*kpoint_weight
        band_index = f_h5["results"]["linear_response"]["bse_index"][0][:, :, :]
        kpoint_coords = f_h5["results"]["electron_eigenvalues"]["kpoint_coords"][:, :]
        band_energies = f_h5["results"]["electron_eigenvalues"]["eigenvalues"][0, :, 0:(band_index.shape[-1] + band_index.shape[-2])]


    # Replicate the control file from vaspout.h5
    ## Replicate the Kx Ky Kz columns
    fbzkpts_col = []
    for i, kpoint in enumerate(kpoint_coords):
        fbzkpts_col += [kpoint] * (len(set(band_index[i,:,:].reshape(1, -1)[0])))
    fbzkpts_col = fbzkpts_col * n_exc

    ## Replicate the E_v E_c nbands_v nbands_c columns
    cond_tot = band_index.shape[2]
    val_num_col = []
    cond_num_col = []
    val_ene_col = []
    cond_ene_col = []
    for k, nkpoint in enumerate(band_index):
        trans_omega_count = 0
        
        for c in range(len(nkpoint)):
            trans_omega_count = len(set(band_index[k, c, :]))
            cond_num_col += [1 + c + cond_tot] * trans_omega_count
            cond_ene_col += [band_energies[k, cond_num_col[-1]-1]] * trans_omega_count


            for v in range(trans_omega_count):
                val_num_col += [1 + v + cond_tot - trans_omega_count]
                val_ene_col += [band_energies[k, val_num_col[-1]-1]]


    val_num_col *= n_exc
    cond_num_col *= n_exc
    val_ene_col *= n_exc
    cond_ene_col *= n_exc


    ## Replicate the Abs(X_BSE)/W_k columns
    rel_exc_strength_col = abs(fatbands[:, 0] + fatbands[:, 1]*(1j))/kpoint_weight

    # Finalise the test Dataframe
    fbzkpts_col = np.array(fbzkpts_col)
    val_ene_col = np.array(val_ene_col).reshape(-1, 1)
    cond_ene_col = np.array(cond_ene_col).reshape(-1, 1)
    rel_exc_strength_col = np.array(rel_exc_strength_col).reshape(-1, 1)

    val_num_col = np.array(val_num_col).reshape(-1, 1)
    cond_num_col = np.array(cond_num_col).reshape(-1, 1)
    fatbands = np.array(fatbands)

    df_test = np.concat((fbzkpts_col, val_ene_col, cond_ene_col, rel_exc_strength_col, val_num_col, cond_num_col, fatbands), axis=1)
    df_test = pl.from_numpy(df_test, schema={"Kx": pl.Float64, "Ky": pl.Float64, "Kz": pl.Float64, "E_v": pl.Float64, "E_c": pl.Float64, "Abs(X_BSE)/W_k": pl.Float64,
                                             "nbands_v": pl.UInt8, "nbands_c": pl.UInt8, "Re(X_BSE)": pl.Float64, "Im(X_BSE)": pl.Float64})

    return df_test
    

def verify(matl_path: str, df_fatbands: pl.DataFrame, verbose: bool = False):
    n_exc = df_fatbands.shape[0]

    # Obtain the control, if it exists
    df_ctrl = []
    i = 1
    with open(matl_path + "/4-BSE/BSEFATBAND", "r") as f_ctrl:
        df_ctrl = []
        # exc_bandgaps2 = []
        for line in f_ctrl:
            row_data = line.split()
            if i > abs(n_exc):
                break

            if len(row_data) == 11:
                df_ctrl.append(row_data)
                i += 1

        df_ctrl = np.array(df_ctrl)
        df_ctrl = np.delete(df_ctrl, 9, axis=1)

        # Get number of rounding decimal places for each column in the ctrl file
        ctrl_rounding_ls = [len(s.split(".")[-1]) for s in df_ctrl[0]]

        # Finalise the ctrl DataFrame
        df_ctrl = pl.from_numpy(df_ctrl, schema={"Kx": pl.Float64, "Ky": pl.Float64, "Kz": pl.Float64, "E_v": pl.Float64, "E_c": pl.Float64,"Abs(X_BSE)/W_k": pl.Float64,
                                                    "nbands_v": pl.UInt8, "nbands_c": pl.UInt8, "Re(X_BSE)": pl.Float64, "Im(X_BSE)": pl.Float64})

    # Verify if df_ctrl is same as df_test
    verify_ls = []
    for i, col in enumerate(df_ctrl.columns):
        if np.array_equal(df_fatbands[col].round(ctrl_rounding_ls[i]),  df_ctrl[col]):
            verify_ls.append((col, True))
        else:
            verify_ls.append((col, False))

    verify_ls = np.array(verify_ls).reshape(-1, 2)
    test_seed = np.random.default_rng().integers(len(df_fatbands), size=10)

    if verbose:
        print("\n--- VERIFICATION RESULTS (RANDOM INDICES SAMPLING): ---")
        print(f"CTRL: {df_ctrl[test_seed]}")
        print(f"CALCULATED: {df_fatbands[test_seed]}\n")
        print(verify_ls)

    return verify_ls
