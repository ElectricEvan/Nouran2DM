from core.exciton_optics import Exciton_Optics
from file_io.tree_h5_dataset import expand_dataset
import pickle

def main():
    # # Material Objectify and Analyse
    # MoS2 = Exciton_Optics("../projects/def-rubel-ac/tanudjae/matls_data/MoS2_ML_rerun", n_exc=10)
    # MoS2.analyse_excitons()
    # MoS2.solve_brightness()
    # MoS2.solve_brightness(light_polar=(1/2**0.5, -1j*1/2**0.5, 0))
    # MoS2.solve_brightness(light_polar=(1/2**0.5, 1j*1/2**0.5, 0))

    # MoSe2 = Exciton_Optics("../projects/def-rubel-ac/tanudjae/matls_data/MoSe2_ML", n_exc=10)
    # MoSe2.analyse_excitons()
    # MoSe2.solve_brightness()
    # MoSe2.solve_brightness(light_polar=(1/2**0.5, -1j*1/2**0.5, 0))
    # MoSe2.solve_brightness(light_polar=(1/2**0.5, 1j*1/2**0.5, 0))

    # WS2 = Exciton_Optics("../projects/def-rubel-ac/tanudjae/matls_data/WS2_ML", n_exc=10)
    # WS2.analyse_excitons()
    # WS2.solve_brightness()
    # WS2.solve_brightness(light_polar=(1/2**0.5, -1j*1/2**0.5, 0))
    # WS2.solve_brightness(light_polar=(1/2**0.5, 1j*1/2**0.5, 0))

    # WSe2 = Exciton_Optics("../projects/def-rubel-ac/tanudjae/matls_data/WSe2_ML", n_exc=10)
    # WSe2.analyse_excitons()
    # WSe2.solve_brightness()
    # WSe2.solve_brightness(light_polar=(1/2**0.5, -1j*1/2**0.5, 0))
    # WSe2.solve_brightness(light_polar=(1/2**0.5, 1j*1/2**0.5, 0))

    # MoTe2 = Exciton_Optics("../projects/def-rubel-ac/tanudjae/matls_data/MoTe2_ML", n_exc=10)
    # MoTe2.analyse_excitons()
    # MoTe2.solve_brightness()
    # MoTe2.solve_brightness(light_polar=(1/2**0.5, -1j*1/2**0.5, 0))
    # MoTe2.solve_brightness(light_polar=(1/2**0.5, 1j*1/2**0.5, 0))


    # # Export Material Object for Quicker Future Runtime
    # with open("MoS2.pkl", "wb") as f:
    #     pickle.dump(MoS2, f)

    # with open("MoSe2.pkl", "wb") as f:
    #     pickle.dump(MoSe2, f)
    
    # with open("WS2.pkl", "wb") as f:
    #     pickle.dump(WS2, f)
    
    # with open("WSe2.pkl", "wb") as f:
    #     pickle.dump(WSe2, f)
    
    # with open("MoTe2.pkl", "wb") as f:
    #     pickle.dump(MoTe2, f)


    # Load Data
    with open("MoS2.pkl", "rb") as f:
        MoS2 = pickle.load(f)

    with open("MoSe2.pkl", "rb") as f:
        MoSe2 = pickle.load(f)
    
    with open("WS2.pkl", "rb") as f:
        WS2 = pickle.load(f)
    
    with open("WSe2.pkl", "rb") as f:
        WSe2 = pickle.load(f)
    
    with open("MoTe2.pkl", "rb") as f:
        MoTe2 = pickle.load(f)
    
    MoTe2.brightness_plot()
    MoTe2.brightness_plot(light_polar=(0.7071067811865475, -0.7071067811865475j, 0))
    MoTe2.brightness_plot(light_polar=(0.7071067811865475, 0.7071067811865475j, 0))
    

if __name__ == "__main__":
    main()