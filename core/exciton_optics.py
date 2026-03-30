import h5py
import numpy as np
import polars as pl
import json
from core.exceptions import FBZKPTsMismatchError
from core.fatbands import Fatbands
from file_io.read_WAVEDER import get_mom_mat

class Exciton_Optics:
    def __init__(self, matl_path, n_exc):
        self.matl_path = matl_path
        self.n_exc = n_exc
        self.excitons = Fatbands(matl_path=self.matl_path, n_exc=self.n_exc)
        self.mom_mat = get_mom_mat(matl_path=self.matl_path)
        self.exc_dipole_vect_dict = {
            "Full k-Space": {},
            "K-Valley": {},
            "Kpr-Valley": {}
        }
        self.brightnesses = {
            "Unpolarised": {
                "Full k-Space": {},
                "K-Valley": {},
                "Kpr-Valley": {}
            }
        }

    def analyse_excitons(self, light_polar: list = 0):
        '''
        Calculates the BSE oscilator strength of an exciton 

        Args:
            matl_path (str): Path to the material folder
            df_excitons (polars.DataFrame): Polars DataFrame of all excitonic transitions for one or multiple excitons
            mom_mat (numpy.ndarray): Numpy array of dipole moment matrix elements in the format of (nband_v, nband_c, kpt_coord, ispin, directional_element)
        
        Returns:
            bse_strengths (list): A list of calculated BSE oscillator strengths
        '''
        df_excitons = self.excitons.df

        # Get relevant data
        with h5py.File(self.matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
            fbzkpts = f_h5["results"]["electron_eigenvalues"]["kpoint_coords"][:]
            n_exc_trans = f_h5["results"]["linear_response"]["bse_fatbands"].shape[1]
            # exc_bandgaps = f_h5["results"]["linear_response"]["opticaltransitions"][0:n_exc, 0]
            
        # Get KPT symmetry mapping, symmetry ops, and relevant data
        with h5py.File(self.matl_path + "/3-GW/vaspout.h5", "r") as f_h5:
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
        df_excitons = df_excitons.with_columns(pl.Series("X_BSE", df_excitons[:, -2].to_numpy() + df_excitons[:, -1].to_numpy()*(1j))).drop(["Re(X_BSE)", "Im(X_BSE)"])
        df_excitons = df_excitons.drop(["E_v", "E_c", "Abs(X_BSE)/W_k"])

        # Replicate the control data
        ## Tabulate rxyz_col and S_col
        rxyz_col = []
        Sxyz_col = []
        fbzkpts = pl.DataFrame(fbzkpts)
        j = 0
        for i in range(len(df_excitons[0:n_exc_trans])):
            if not np.array_equal(df_excitons[i, 0:3], fbzkpts[j]):
                j += 1
            Sxyz = lat_vect.T @ kpt_symops[j] @ np.linalg.inv(lat_vect.T)
            rxyz = Sxyz @ self.mom_mat[df_excitons[i, 3]-1, df_excitons[i, 4]-1, kpt_mapping[j]-1][0]
            Sxyz_col.append(Sxyz)
            rxyz_col.append(rxyz)

        rxyz_col *= self.n_exc
        rxyz_col = np.array(rxyz_col)
        df_excitons = df_excitons.with_columns(pl.Series("rx", rxyz_col[:, 0]))
        df_excitons = df_excitons.with_columns(pl.Series("ry", rxyz_col[:, 1]))
        df_excitons = df_excitons.with_columns(pl.Series("rz", rxyz_col[:, 2]))
        df_excitons = df_excitons.with_columns(pl.col(["Kx", "Ky", "Kz"]).round(6)) # Rounding kpoint coords to filter easier


        ## Calculate the BSE strengths
        bse_strengths_test = []
        if light_polar:
            polar_strength = []
        # l_polar_strength = []
        # non_polar_strength = []
        # non_polar_strength_norm = []
        # planar_strength = []

        v_min = df_excitons["nbands_v"].min()
        v_max = df_excitons["nbands_v"].max()
        c_min = df_excitons["nbands_c"].min()
        c_max = df_excitons["nbands_c"].max()

        self.exc_dipole_vect_ls = {}
        for i in range(self.n_exc):
            exc_dipole_vect = []; exc_dipole_vect_K = []; exc_dipole_vect_Kpr = []
            for a in range(rxyz_col.shape[1]): # a = x, y, z
                exc_dipole_vect_a = 0 ; exc_dipole_vect_a = 0; exc_dipole_vect_a_K = 0; exc_dipole_vect_a_Kpr = 0 
                for v in range(v_min, v_max+1):
                    for c in range(c_min, c_max+1):
                        # Sum all c, v for all kpoints by filtering for c and v in the df
                        df_excitons_bands_filtered = df_excitons[i*n_exc_trans:(i+1)*n_exc_trans].filter((pl.col("nbands_v") == v) & (pl.col("nbands_c") == c))    # Sum Ks
                        df_excitons_bands_filtered_K = df_excitons_bands_filtered.filter((pl.col("Kx") == 0.333333) & (pl.col("Ky") == 0.333333) & (pl.col("Kz") == 0))
                        df_excitons_bands_filtered_Kpr = df_excitons_bands_filtered.filter((pl.col("Kx") == -0.333333) & (pl.col("Ky") == 0.666667) & (pl.col("Kz") == 0))
                        df_excitons_bands_filtered = df_excitons_bands_filtered[:, -4:].to_numpy()
                        df_excitons_bands_filtered_K = df_excitons_bands_filtered_K[:, -4:].to_numpy()
                        df_excitons_bands_filtered_Kpr = df_excitons_bands_filtered_Kpr[:, -4:].to_numpy()
                        exc_dipole_vect_a += np.sum(df_excitons_bands_filtered[:, 1+a] * (df_excitons_bands_filtered[:, 0])) # matrix element * bse vector
                        exc_dipole_vect_a_K += np.sum(df_excitons_bands_filtered_K[:, 1+a] * (df_excitons_bands_filtered_K[:, 0])) # matrix element * bse vector
                        exc_dipole_vect_a_Kpr += np.sum(df_excitons_bands_filtered_Kpr[:, 1+a] * (df_excitons_bands_filtered_Kpr[:, 0])) # matrix element * bse vector


                exc_dipole_vect.append(exc_dipole_vect_a)
                exc_dipole_vect_K.append(exc_dipole_vect_a_K)
                exc_dipole_vect_Kpr.append(exc_dipole_vect_a_Kpr)
            
            self.exc_dipole_vect_dict["Full k-Space"][i] = exc_dipole_vect
            self.exc_dipole_vect_dict["K-Valley"][i] = exc_dipole_vect_K
            self.exc_dipole_vect_dict["Kpr-Valley"][i] = exc_dipole_vect_Kpr
            # bse_strengths_test.append(float(sum([abs(comp**2) for comp in exc_dipole_vect])))
            # l_light_polar = np.array([1/2**0.5, -1j*1/2**0.5, 0])
            # non_polar = np.array([1, 0, 0])
            # non_polar_norm = np.array([1/3, 1/3, 1/3])
            # planar = np.array([1/2**0.5, 1/2**0.5, 0])
            # if light_polar:
            #    polar_strength.append(float(abs(light_polar @ np.array(exc_dipole_vect)**2)))
            # l_polar_strength.append(float(abs(l_light_polar @ np.array(exc_dipole_vect)**2)))
            # non_polar_strength.append(float(abs(non_polar @ np.array(exc_dipole_vect)**2)))
            # non_polar_strength_norm.append(float(abs(non_polar_norm @ np.array(exc_dipole_vect)**2)))
            # planar_strength.append((float(abs(planar @ np.array(exc_dipole_vect)**2))))

        # print(f"{r_polar_strength=}")
        # print(f"{l_polar_strength=}")
        # print(f"{non_polar_strength=}")
        # print(f"{non_polar_strength_norm=}")
        # print(f"{planar_strength=}")

        print(f"\nBRIGHTNESSES OF {self.n_exc} LOWEST ENERGY EXCITONS: {bse_strengths_test}")
        if light_polar:
            print(f"\nBRIGHTNESSES OF {self.n_exc} LOWEST ENERGY EXCITONS: {self.brightnesses}")


    def solve_brightness(self, light_polar: list = None):
        if light_polar:
            light_polar_mat = np.array(light_polar)
            light_polar = str(light_polar).strip("[,]")
            self.brightnesses[light_polar] = {
                "Full k-Space": {},
                "K-Valley": {},
                "Kpr-Valley": {}
            }

        for i in range(self.n_exc):
            if len(self.brightnesses["Unpolarised"]["Full k-Space"]) < self.n_exc:
                self.brightnesses["Unpolarised"]["Full k-Space"][i] = float(sum([abs(comp**2) for comp in self.exc_dipole_vect_dict["Full k-Space"][i]]))
                self.brightnesses["Unpolarised"]["K-Valley"][i] = float(sum([abs(comp**2) for comp in self.exc_dipole_vect_dict["K-Valley"][i]]))
                self.brightnesses["Unpolarised"]["Kpr-Valley"][i] = float(sum([abs(comp**2) for comp in self.exc_dipole_vect_dict["Kpr-Valley"][i]]))

            if light_polar:
                self.brightnesses[light_polar]["Full k-Space"][i] = float(abs(light_polar_mat @ np.array(self.exc_dipole_vect_dict["Full k-Space"][i])**2))
                self.brightnesses[light_polar]["K-Valley"][i] = float(abs(light_polar_mat @ np.array(self.exc_dipole_vect_dict["K-Valley"][i])**2))
                self.brightnesses[light_polar]["Kpr-Valley"][i] = float(abs(light_polar_mat @ np.array(self.exc_dipole_vect_dict["Kpr-Valley"][i])**2))
        

    def verify_brightness(self, verbose: bool = False):
        '''
        Reads the BSE oscillator strength from vaspout.h5 as a control to verify the this library's results to it, and prints out the verification results.

        Args:
            matl_path (str): Path to the material folder
            bse_strengths (list): 
        a list containing boolean expressions for if \
        the calculated BSE oscillator strengths match the control. None if False.
        '''
        # Get relevant data
        with h5py.File(self.matl_path + "/4-BSE/vaspout.h5", "r") as f_h5:
            bse_strengths_ctrl = f_h5["results"]["linear_response"]["opticaltransitions"][0:self.n_exc, 1]

        bse_strengths_test = list(self.brightnesses["Unpolarised"]["Full k-Space"].values())
        verify_ls = np.isclose(bse_strengths_ctrl, bse_strengths_test, rtol=1E-13, atol=1E-13).tolist()

        if verbose:
            print("\n--- VERIFICATION RESULTS: ---")
            print(f"CTRL: {bse_strengths_ctrl.tolist()}")
            print(f"CALCULATED: {bse_strengths_test}\n")
            print(verify_ls)

        return verify_ls
    
