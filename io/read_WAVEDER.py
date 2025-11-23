import h5py
import numpy as np
from pymatgen.io.vasp import Poscar
import spglib


# f_wave = open(r"4-BSE/WAVEDER", "rb")
# prefix = np.fromfile(f_wave, dtype=np.int32, count=1)[0]
# [nbands, nbcder, nktot, nstot] = np.fromfile(f_wave, dtype=np.int32, count=4)
# suffix = np.fromfile(f_wave, dtype=np.int32, count=1)[0]
# if(abs(suffix)-abs(prefix)):
#   print("Read incorrect number of bytes")
#   print("Expected: ", prefix, "Read: 32")

with open(r"MoS2_ML/2-DFT-empty/WAVEDER", "rb") as fp:
  # filecontent = fp.read()
  # Read n-bands_tot, n-bands_cder, n-kpts, i-spin
  prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  [nb, nbcder, nktot, nstot] = np.fromfile(fp, dtype=np.int32, count=4)
  suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  if(abs(suffix)-abs(prefix)):
    print("Read incorrect number of bytes")
    print("Expected: ", prefix, "Read: 32")

  # nodes_in_dielectric_function - not used
  prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  nodes_in_dielectric_function = np.fromfile(fp, dtype=np.float64, count=1)
  suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  if(abs(suffix)-abs(prefix)):
    print("Read incorrect number of bytes")
    print("Expected: ", prefix, "Read: 8")

  # WPLASMON - not used
  prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  WPLASMON = np.fromfile(fp, dtype=np.float64, count=9)
  suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  if(abs(suffix)-abs(prefix)):
    print("Read incorrect number of bytes")
    print("Expected: ", prefix, "Read: 72")

  # Read Matrix Elements (rijks)
  list_ = []
  rijks_data = np.array([])
  prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  list_.append(np.fromfile(fp, dtype=np.complex64, count=3*nb*nbcder*nktot*nstot))
  suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
  if(abs(suffix)-abs(prefix)):
    print("Read incorrect number of bytes")
    print("Expected: ", prefix, "Read: ", 3*nb*nbcder*nstot*nktot*8)


# Reshape the matrix elements to the appropriate size
rijks_data = np.array(list_)
# print(np.shape(rijks_data)) # Check 

rijks = rijks_data.reshape(3, nstot, nktot, nbcder, nb).T

# poscar = Poscar.from_file(r"2-DFT-empty/POSCAR")
# struct = poscar.structure
# lat = struct.lattice.matrix
# pos = struct.frac_coords.tolist()
# at_nums = [site.specie.number for site in struct]
# sym_data = spglib.get_symmetry_dataset((lat, pos, at_nums), symprec=1e-6)

print(rijks)