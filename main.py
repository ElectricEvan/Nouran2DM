from core import exciton_optics
from core import fatbands
from file_io import read_WAVEDER

def main():
    matl_path = "../projects/def-rubel-ac/tanudjae/matls_data/MoSe2_ML"
    df_fatbands = fatbands.valid_transitions(matl_path=matl_path, n_exc=-10)
    fatbands.verify(matl_path=matl_path, df_fatbands=df_fatbands, verbose=True)
    mom_mat = read_WAVEDER.get_mom_mat(matl_path=matl_path)
    bse_strengths = exciton_optics.exciton_optics(matl_path=matl_path, df_fatbands=df_fatbands, mom_mat=mom_mat)
    exciton_optics.verify(matl_path=matl_path, bse_strengths=bse_strengths, verbose=True)
    

if __name__ == "__main__":
    main()