import h5py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from random import choice
import data.util as util

import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)
    
def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std
    
    
class DataTransform2:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """

    def __call__(self, image, target, fat_supp, opt):#, att):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # Normalize input
        image = to_tensor(image)
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = to_tensor(target)
        # Normalize target
        target = normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)

        image = image.unsqueeze(0)
        target = target.unsqueeze(0)
        
        if opt['phase'] == 'train':
            _, H, W = image.shape
            HR_size = opt['HR_size']
            
            # randomly crop
            h = random.randint(0, max(0, H - HR_size))
            w = random.randint(0, max(0, W - HR_size))
            image = image[:, h:h + HR_size, w:w + HR_size]
            target = target[:, h:h + HR_size, w:w + HR_size]

        return image, target, mean, std, fat_supp#attrs['norm'].astype(np.float32)#
    

class HDF5Dataset(Dataset):
    def __init__(self, opt, transform=DataTransform2(), acc_fac=None):
        super(HDF5Dataset, self).__init__()

        self.opt = opt
        self.acc_fac = acc_fac
        self.transform = transform
        self.paths_h5 = None
        self.data_type = 'h5'

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        _, self.paths_h5 = util.get_image_paths(self.data_type, opt['dataroot_HR'])
        file_names = self.paths_h5
        data_path = Path(opt['dataroot_HR'])
        #print(f'Initializing {data_path.stem}. This might take a minute')
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
        #print(f'Finished {data_path.stem} initialization!')

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
                

        return ds_slice_arr, gt_slice_arr, fat_supp#, hf.attrs

    # TODO: Must turn outputs into tensors. They are currently numpy arrays or python booleans.
    def __getitem__(self, idx):  # Need to add transforms.
        file_name, slice_num, acc_fac = self.names_and_slices[idx]
        ds_slice, gt_slice, fat_supp = self.h5_slice_parse_fn(file_name, slice_num, acc_fac)
        image, target, mean, std, fat_supp = self.transform(ds_slice, gt_slice, fat_supp, self.opt)
        return {'LR': image, 'HR': target, 'LR_path': self.paths_h5, 'HR_path': self.paths_h5, 'Mean': mean, 'Std': std, 'Ect': fat_supp}#, att)  # Type is ndarray float32, ndarray float32, and python boolean respectively
        
    
    