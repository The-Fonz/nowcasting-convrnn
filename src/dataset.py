import glob, os

import numpy as np
import h5py

import utils


class ReflCompDataset():
    def __init__(self, folder):
        "Search folder for hdf5 files and add them to dataset"
        # Sort by filename, which reflects date
        self.h5files = sorted(glob.glob(os.path.join(folder, '*.h5')))
    
    def show_structure(self, idx):
        "Returns list of lines, showing groups, datasets and attributes"
        h5file = self.get_file(idx)
        return utils.explain_hdf5_file(h5file)
    
    def __getitem__(self, idx):
        "Get img data array"
        # Use with block to not leave h5py file open, otherwise we hit
        # max open file limit
        with h5py.File(self.h5files[idx], 'r') as f:
            # Return as float always to avoid problems later on
            return np.array(f['image1/image_data']).astype(np.float32)
    
    def get_file(self, idx):
        "Get h5py file"
        return h5py.File(self.h5files[idx], 'r')

    def crunch(n_workers=8):
        "Crunch through the entire dataset, gathering stats"
        pass
