import h5py
import numpy as np

with h5py.File(r"MoS2_ML/1-DFT/vaspout.h5", "r") as f:
    for keyL1 in f:
        print(f"{keyL1}:")
        try:
            for keyL2 in f[keyL1]:
                print(f"|----{keyL2}:")
                try:
                    for keyL3 in f[keyL1][keyL2]:
                        print(f"|    |----{keyL3}:")
                        try:
                            for keyL4 in f[keyL1][keyL2][keyL3]:
                                print(f"|    |    |----{keyL4}:")
                        except TypeError:
                            pass
                except TypeError:
                    pass
        except TypeError:
            pass