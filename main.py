import h5py
import numpy as np
import polars as pl
import os

class ExcitonRangeError:
    def __init__(self, message="n_exc out of range from what VASP calculated"):
        self.message = message
        super().__init__(self.message)


class FBZKPTsMismatchError:
    def __init__(self, message="The FBZKPTs from BSE dataset is different from the GW dataset"):
        self.message = message
        super().__init__(self.message)

def valid_transitions(BSE_path: str, verify: bool=False, n_exc: int=0):
    """
    Gets all excitonic transitions from BSE vaspout.h5 that are below Omega_max (max energy difference for excitation pairs). These transistions include all the various
    points (Kx, Ky, Kz), energy levels the valence and conduction bands (E_v, E_c) within Omega_max difference, the relative strength of the exciton's component at that 
    kpoint (Abs(X_BSE)/W_k), the valence and conduction band orbital numbers (nbands_v, nbands_c), and the complex amplitude of the e-h bound state (Re(X_BSE), Im(X_BSE)). 
    Although, this information is provided by BSEFATBAND, it loses precision from rounding values and may not always be included.

    Args:
        BSE_path (str): Path to BSE folder
        verify (bool): If True, opens BSEFATBAND as a control to verify if this library's results matches with it, and prints a sample of results from random indices. \
            Defaults to False. 
        n_exc (int): Singles out the data for a single excitonic bandgap energy where "1" is the lowest. Using negative numbers takes the range of that number of \
            excitonic bangaps starting from the lowest value. Defaults to 1

    Returns:
        (df_trans, verfify_ls) (tuple): Containing a DataFrame of all excitonic transitions' kpoint coordinates, valence/conduction band index, and complex BSE amplitude \
            obtained from vaspout.h5 and, if verify is True, a numpy.ndarray containing tuples of every column in BSEFATBANDS and boolean expressions for if that column \
                replicates the control file correctly. None if False
    """

    # Open all relevant files
    with h5py.File(BSE_path + "/vaspout.h5", "r") as f_h5:
        kpoint_weight = f_h5["results"]["electron_eigenvalues"]["kpoints_symmetry_weight"][0]
        exc_count = f_h5["results"]["linear_response"]["bse_fatbands"].shape[0]
        n_exc_trans = f_h5["results"]["linear_response"]["bse_fatbands"].shape[1]

        # Check if n_exc is within a valid range
        if abs(n_exc) > exc_count:
            raise ExcitonRangeError(f"{abs(n_exc)} n_exc is out of range for VASP calculations which only contains {exc_count} excitons.")

        fatbands = f_h5["results"]["linear_response"]["bse_fatbands"][0:abs(n_exc), :, :].reshape(-1,2)*kpoint_weight
        band_index = f_h5["results"]["linear_response"]["bse_index"][0][:, :, :]
        kpoint_coords = f_h5["results"]["electron_eigenvalues"]["kpoint_coords"][:, :]
        band_energies = f_h5["results"]["electron_eigenvalues"]["eigenvalues"][0, :, 0:(band_index.shape[-1] + band_index.shape[-2])]

    # Create the control file from BSEFATBAND first
    if verify:
        df_ctrl = []
        i = 1
        with open(BSE_path + "/BSEFATBAND", "r") as f_ctrl:
            df_ctrl = []
            # exc_bandgaps2 = []
            for line in f_ctrl:
                row_data = line.split()
                if i > abs(n_exc)*n_exc_trans:
                    break

                if len(row_data) == 11:
                    df_ctrl.append(row_data)
                    i += 1

                # elif len(row_data) == 5:
                #     if len(exc_bandgaps2) == abs(n_exc):
                #         break
                #     exc_bandgaps2.append(row_data[2])

        df_ctrl = np.array(df_ctrl)
        df_ctrl = np.delete(df_ctrl, 9, axis=1)

        # Get number of rounding decimal places for each column in the ctrl file
        ctrl_rounding_ls = [len(s.split(".")[-1]) for s in df_ctrl[0]]

        # Finalise the ctrl DataFrame
        df_ctrl = pl.from_numpy(df_ctrl, schema={"Kx": pl.Float64, "Ky": pl.Float64, "Kz": pl.Float64, "E_v": pl.Float64, "E_c": pl.Float64,"Abs(X_BSE)/W_k": pl.Float64,
                                                 "nbands_v": pl.UInt8, "nbands_c": pl.UInt8, "Re(X_BSE)": pl.Float64, "Im(X_BSE)": pl.Float64})
    
    # Replicate the control file from vaspout.h5
    ## Replicate the Kx Ky Kz columns
    fbzkpts_col = []
    for i, kpoint in enumerate(kpoint_coords):
        fbzkpts_col += [kpoint] * (len(set(band_index[i,:,:].reshape(1, -1)[0])))
    fbzkpts_col = fbzkpts_col * abs(n_exc)

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
            if verify:
                cond_ene_col += [band_energies[k, cond_num_col[-1]-1]] * trans_omega_count


            for v in range(trans_omega_count):
                val_num_col += [1 + v + cond_tot - trans_omega_count]
                if verify:
                    val_ene_col += [band_energies[k, val_num_col[-1]-1]]


    val_num_col *= abs(n_exc)
    cond_num_col *= abs(n_exc)
    val_ene_col *= abs(n_exc)
    cond_ene_col *= abs(n_exc)


    ## Replicate the Abs(X_BSE)/W_k columns
    rel_exc_strength_col = abs(fatbands[:, 0] + fatbands[:, 1]*(1j))/kpoint_weight

    # Finalise the test Dataframe
    fbzkpts_col = np.array(fbzkpts_col)
    if verify:
        val_ene_col = np.array(val_ene_col).reshape(-1, 1)
        cond_ene_col = np.array(cond_ene_col).reshape(-1, 1)
        rel_exc_strength_col = np.array(rel_exc_strength_col).reshape(-1, 1)

    val_num_col = np.array(val_num_col).reshape(-1, 1)
    cond_num_col = np.array(cond_num_col).reshape(-1, 1)
    fatbands = np.array(fatbands)
    if verify: 
        df_test = np.concat((fbzkpts_col, val_ene_col, cond_ene_col, rel_exc_strength_col, val_num_col, cond_num_col, fatbands), axis=1)
        df_test = pl.from_numpy(df_test, schema={"Kx": pl.Float64, "Ky": pl.Float64, "Kz": pl.Float64, "E_v": pl.Float64, "E_c": pl.Float64, "Abs(X_BSE)/W_k": pl.Float64,
                                                "nbands_v": pl.UInt8, "nbands_c": pl.UInt8, "Re(X_BSE)": pl.Float64, "Im(X_BSE)": pl.Float64})
    else:
        df_test = np.concat((fbzkpts_col, val_num_col, cond_num_col, fatbands), axis=1)
        df_test = pl.from_numpy(df_test, schema={"Kx": pl.Float64, "Ky": pl.Float64, "Kz": pl.Float64, "nbands_v": pl.UInt8, "nbands_c": pl.UInt8,
                                                 "Re(X_BSE)": pl.Float64, "Im(X_BSE)": pl.Float64})
        
    # Single out data for one exciton if the user selected the option
    if n_exc > 0:
        df_test = df_test[(n_exc-1)*n_exc_trans:n_exc*n_exc_trans]
        df_ctrl = df_ctrl[(n_exc-1)*n_exc_trans:n_exc*n_exc_trans]

    # Verify if df_ctrl is same as df_test
    if verify:
        verify_ls = []
        for i, col in enumerate(df_ctrl.columns):
            if np.array_equal(df_test[col].round(ctrl_rounding_ls[i]),  df_ctrl[col]):
                verify_ls.append((col, True))
            else:
                verify_ls.append((col, False))

        verify_ls = np.array(verify_ls).reshape(-1, 2)
        test_seed = np.random.default_rng().integers(len(df_test), size=10)
        print("\n--- VERIFICATION RESULTS (RANDOM INDICES SAMPLING): ---")
        print(f"CTRL: {df_ctrl[test_seed]}")
        print(f"CALCULATED: {df_test[test_seed]}\n")

        df_test = df_test.with_columns(pl.Series("X_BSE", df_test[:, -2].to_numpy() + df_test[:, -1].to_numpy()*(1j))).drop(["Re(X_BSE)", "Im(X_BSE)"])
        return df_test[:, [0, 1, 2, 6, 7, 8]], verify_ls
    else:
        df_test = df_test.with_columns(pl.Series("X_BSE", df_test[:, -2].to_numpy() + df_test[:, -1].to_numpy()*(1j))).drop(["Re(X_BSE)", "Im(X_BSE)"])
        return df_test, None


def get_mom_mat(BSE_path: str):
    """
    Obtains the complex dipole moment matrix elements and shortens the list based on the number of particpating bands for excitonic transitions.
    
    Args:
        BSE_path (str): Path to BSE folder

    Returns:
        mom_mat (numpy.ndarray): Array of dipole moment matrix elements in the format of (nband_v, nband_c, kpt_coord, ispin, directional_element)
    """
    with open(BSE_path + "/WAVEDER", "rb") as fp:
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

    with h5py.File(BSE_path + "/vaspout.h5", "r") as f_h5:
        val_num_max = f_h5["results"]["linear_response"]["bse_index"].shape[3]
        cond_num_max = val_num_max + f_h5["results"]["linear_response"]["bse_index"].shape[2]

    return mom_mat[0:val_num_max, 0:cond_num_max, :, :, :]


def exciton_optics(BSE_path: str, GW_path: str, df: pl.DataFrame, mom_mat: np.ndarray, verify: bool=False):
    '''
    Calculates the BSE oscilator strength of an exciton 

    Args:
        BSE_path (str): Path to BSE folder
        GW_path (str): Path to GW folder
        df (polars.DataFrame): Polars DataFrame of all excitonic transitions for one or multiple excitons
        mom_mat (numpy.ndarray): Numpy array of dipole moment matrix elements in the format of (nband_v, nband_c, kpt_coord, ispin, directional_element)
        verify (bool): If True, reads the BSE oscillator strength from vaspout.h5 as a control to verify the this library's results to it, and prints out the \
            verification results. Defaults to False
    
    Returns:
        (bse_strengths, verify_ls) (tuple): Containing a list of calculated BSE oscillator strengths and, if verify True, a list containing boolean expressions for if \
        the calculated BSE oscillator strengths match the control. None if False.
    '''
    # Get relevant data
    with h5py.File(BSE_path + "/vaspout.h5", "r") as f_h5:
        fbzkpts = f_h5["results"]["electron_eigenvalues"]["kpoint_coords"][:]
        n_exc_trans = f_h5["results"]["linear_response"]["bse_fatbands"].shape[1]
        n_exc = int(len(df)/n_exc_trans)
        # exc_bandgaps = f_h5["results"]["linear_response"]["opticaltransitions"][0:n_exc, 0]
        if verify:
            bse_strengths = f_h5["results"]["linear_response"]["opticaltransitions"][0:n_exc, 1]
        
    # Get KPT symmetry mapping, symmetry ops, and relevant data
    with h5py.File(GW_path + "/vaspout.h5", "r") as f_h5:
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


    # Replicate the control data
    ## Tabulate rxyz_col and S_col
    rxyz_col = []
    Sxyz_col = []
    fbzkpts = pl.DataFrame(fbzkpts)
    j = 0
    for i in range(len(df[0:n_exc_trans])):
        if not np.array_equal(df[i, 0:3], fbzkpts[j]):
            j += 1
        Sxyz = lat_vect.T @ kpt_symops[j] @ np.linalg.inv(lat_vect.T)
        rxyz = Sxyz @ mom_mat[df[i, 3]-1, df[i, 4]-1, kpt_mapping[j]-1][0]
        Sxyz_col.append(Sxyz)
        rxyz_col.append(rxyz)

    rxyz_col *= n_exc
    rxyz_col = np.array(rxyz_col)
    df = df.with_columns(pl.Series("rx", rxyz_col[:, 0]))
    df = df.with_columns(pl.Series("ry", rxyz_col[:, 1]))
    df = df.with_columns(pl.Series("rz", rxyz_col[:, 2]))

    ## Calculate the BSE strengths
    bse_strengths_test = []
    for i in range(n_exc):
        bse_strength_a = []
        for a in range(rxyz_col.shape[1]): # a = x, y, z
            bse_strength = 0
            for v in range(df["nbands_v"].min(), df["nbands_v"].max()+1):
                for c in range(df["nbands_c"].min(), df["nbands_c"].max()+1):
                    df_bands_filtered = df[i*n_exc_trans:(i+1)*n_exc_trans].filter((pl.col("nbands_v") == v) & (pl.col("nbands_c") == c))[:, -4:].to_numpy()    # Sum Ks
                    bse_strength += np.sum(df_bands_filtered[:, 1+a] * (df_bands_filtered[:, 0])) # Sum i,j, then finally a

            bse_strength_a.append(abs(bse_strength)**2)

        bse_strengths_test.append(float(sum(bse_strength_a)))

    # Verification
    if verify:
        verify_ls = np.isclose(bse_strengths, bse_strengths_test, rtol=1E-13, atol=1E-13).tolist()
        print("\n--- VERIFICATION RESULTS: ---")
        print(f"CTRL: {bse_strengths.tolist()}")
        print(f"CALCULATED: {bse_strengths_test}\n")

        return bse_strengths_test, verify_ls
    else:
        return bse_strengths_test, None    
    

def main():
    df_trans, verify_ls = valid_transitions(BSE_path=r"../matls_data/MoS2_ML/4-BSE", verify=True, n_exc=-10)
    mom_mat = get_mom_mat(BSE_path=r"../matls_data/MoS2_ML/4-BSE")
    print(verify_ls)
    bse_strengths, verify_ls = exciton_optics(BSE_path=r"../matls_data/MoS2_ML/4-BSE", GW_path=r"../matls_data/MoS2_ML/3-GW", df=df_trans, mom_mat=mom_mat, verify=True)
    print(verify_ls)


if __name__ == "__main__":
    main()