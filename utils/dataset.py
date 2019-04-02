import h5py
from torch.utils.data import Dataset
from pathlib import Path
from random import choice
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, data_dir, batch_size=16, training=True, acc_fac=None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.training = training
        self.acc_fac = acc_fac

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        data_path = Path(self.data_dir)
        file_names = [str(h5) for h5 in data_path.glob('*.h5')]
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        print(f'Initializing {data_path.stem}. This might take a minute')
        slice_counts = [self.get_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()

        if self.acc_fac is not None:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, self.acc_fac] for s_idx in range(slice_num)]

        else:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, choice((4, 8))] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            try:  # Train and Val
                return hf['1'].shape[0]
            except KeyError:  # Test
                return hf['data'].shape[0]

    @staticmethod
    def h5_slice_parse_fn(file_name, slice_num, acc_fac):
        with h5py.File(file_name, 'r', libver='latest', swmr=True) as hf:
            ds_slice_arr = np.asarray(hf[str(acc_fac)][slice_num])
            gt_slice_arr = np.asarray(hf['1'][slice_num])
            # Fat suppression ('CORPDFS_FBK', 'CORPD_FBK').
            fat = hf.attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')

        return ds_slice_arr, gt_slice_arr, fat_supp

    # TODO: Must turn outputs into tensors. They are currently numpy arrays or python booleans.
    def __getitem__(self, idx):  # Need to add transforms.
        file_name, slice_num, acc_fac = self.names_and_slices[idx]
        ds_slice, gt_slice, fat_supp = self.h5_slice_parse_fn(file_name, slice_num, acc_fac)
        return ds_slice, gt_slice, fat_supp  # Type is ndarray float32, ndarray float32, and python boolean respectively
