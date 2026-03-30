from core.exciton_optics import Exciton_Optics
from file_io.tree_h5_dataset import expand_dataset

def main():
    MoS2 = Exciton_Optics(matl_path="../projects/def-rubel-ac/tanudjae/matls_data/MoS2_ML_rerun", n_exc=10)
    print()

    

if __name__ == "__main__":
    main()