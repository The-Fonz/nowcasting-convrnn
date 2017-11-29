import glob, os

import h5py

import utils


class ReflCompDataset():
    def __init__(self, folder):
        "Search folder for hdf5 files and add them to dataset"
        # Sort by filename, which reflects date
        self.h5files = sorted(glob.glob(os.path.join(folder, '*.h5')))
    
    def show_structure(self, idx):
        "Returns list of lines, showing groups, datasets and attributes"
        h5file = self.__getitem__(idx)
        return utils.explain_hdf5_file(h5file)
    
    def __getitem__(self, idx):
        return h5py.File(self.h5files[idx], 'r')
    
    def crunch(n_workers=8):
        "Crunch through the entire dataset, gathering stats"
        pass
