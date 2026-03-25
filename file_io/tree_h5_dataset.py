import h5py
import numpy

def expand_dataset(h5_path: str, file_out_path: str):
    with h5py.File(h5_path, "r") as f:
        with open(file_out_path, "w") as out:
            for keyL1 in f:
                out.write(f"{keyL1}:\n")
                try:
                    for keyL2 in f[keyL1]:
                        if isinstance(keyL2, numpy.ndarray) and len(keyL2.shape) > 1:
                            out.write(f"|----[{str(keyL2[0]).replace("\n","")}\n")
                            for line in keyL2[1:-1]:
                                out.write(f"|     {str(line).replace("\n","")}\n")

                            out.write(f"|     {str(line).replace("\n","")}]\n")

                        else:
                            out.write(f"|----{keyL2}:\n")
                            
                        try:
                            for keyL3 in f[keyL1][keyL2]:
                                if isinstance(keyL3, numpy.ndarray) and len(keyL3.shape) > 1:
                                    out.write(f"|    |----[{str(keyL3[0]).replace("\n","")}\n")
                                    for line in keyL3[1:-1]:
                                        out.write(f"|    |     {str(line).replace("\n","")}\n")

                                    out.write(f"|    |     {str(line).replace("\n","")}]\n")

                                else:
                                    out.write(f"|    |----{keyL3}:\n")

                                try:
                                    for keyL4 in f[keyL1][keyL2][keyL3]:
                                        if isinstance(keyL4, numpy.ndarray) and len(keyL4.shape) > 1:
                                            out.write(f"|    |    |----[{str(keyL4[0]).replace("\n","")}\n")
                                            for line in keyL4[1:-1]:
                                                out.write(f"|    |    |     {str(line).replace("\n","")}\n")

                                            out.write(f"|    |    |     {str(line).replace("\n","")}]\n")

                                        else:
                                            out.write(f"|    |    |----{keyL4}:\n")

                                except TypeError:
                                    pass

                        except TypeError:
                            pass

                except TypeError:
                    pass


