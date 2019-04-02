import h5py
import numpy as np
from pathlib import Path


def convert(data_dir, save_dir='new_data'):
    """
    Converts HDF5 files so that they are chunked into slices and have no compression.
    I have found that this is fastest, at least on a SSD.

    :param data_dir: Directory where original (bad) data resides.
    :param save_dir: Folder to save new (faster) data. Do not change for train/val/test
    :return: Saves the new data.
    """

    data_path = Path(data_dir)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    save_path = save_path / data_path.stem
    save_path.mkdir(exist_ok=True)

    for idx, file_path in enumerate(data_path.glob('*.h5'), start=1):
        new_file_path = save_path / (file_path.stem + '.h5')
        print(f'{idx:03d}: {new_file_path}')
        with h5py.File(file_path, mode='r') as hf, h5py.File(new_file_path, mode='x', libver='latest') as new_hf:
            new_hf.attrs.update(dict(hf.attrs))
            for key, value in hf.items():
                if value.shape[1:] == (320, 320):
                    new_dataset = new_hf.create_dataset(name=key, data=np.asarray(value), chunks=(1, 320, 320))
                    new_dataset.attrs.update(dict(hf[key].attrs))
                else:
                    new_dataset = new_hf.create_dataset(name=key, data=np.asarray(value), chunks=True)
                    new_dataset.attrs.update(dict(hf[key].attrs))


if __name__ == '__main__':
    pass
    # Example usage
    # train_dir = '/home/veritas/PycharmProjects/MyFastMRI/old_data/multicoil_train'
    # val_dir = '/home/veritas/PycharmProjects/MyFastMRI/old_data/multicoil_val'
    # test_dir = '/home/veritas/PycharmProjects/MyFastMRI/old_data/multicoil_test'
    #
    # convert(train_dir)
    # convert(val_dir)
    # convert(test_dir)

