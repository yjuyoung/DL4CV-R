import h5py


def get_slice_number(file_name):
    with h5py.File(name=file_name, mode='r') as hf:
        return hf['1'].shape[0]


def get_test_slice_number(file_name):
    with h5py.File(name=file_name, mode='r') as hf:
        return hf['data'].shape[0]
