import os

import mne
import numpy as np
from scipy.io import loadmat


fs = 250
tgt_fs = 250
labels = ['left', 'right', 'foot', 'tongue']
ch_names = [
                              'Fz',
                  'E2', 'E3', 'E4', 'E5', 'E6',
            'E7', 'C3', 'E9', 'Cz', 'E11', 'C4', 'E13',
                'E14', 'E15', 'E16', 'E17', 'E18',
                       'E19', 'Pz', 'E21',
                              'E22'
        ]
ch_types = ['eeg'] * len(ch_names)


root_path = '/root/EEG-FM-Bench/assets/data/raw/BCI'
out_path = '/root/EEG-FM-Bench/assets/data/raw/BCI Competition IV/2a/set'
os.makedirs(out_path, exist_ok=True)


def get_start_index(file_name: str) -> int:
    # A04T contains one leading EOG segment and six valid runs.
    if file_name == 'A04T.mat':
        return 1
    return 3


for f in os.listdir(root_path):
    if not f.endswith('.mat'):
        continue

    struct = loadmat(os.path.join(root_path, f), simplify_cells=True)

    data = struct['data']
    num = len(data)
    start_idx = get_start_index(f)

    print(f'file: {f}, length: {num}, start_idx: {start_idx}')
    for j in range(start_idx, num):
        x = data[j]['X'][:, :22]
        x = x.transpose(1, 0)

        y = data[j]['y'] - 1
        onset_sample = data[j]['trial']
        onset = (onset_sample / fs) + 2.0

        age = data[j]['age']
        gender = data[j]['gender']

        info = mne.create_info(
            ch_names=ch_names,
            ch_types=ch_types,
            sfreq=fs,
        )

        pad = np.zeros(len(y), dtype=np.float32)
        desc = [labels[i] for i in y]

        x = x * 1e-6
        raw = mne.io.RawArray(x, info)

        annotations = mne.Annotations(
            onset=onset,
            duration=pad,
            description=desc
        )

        raw.set_annotations(annotations)
        print(annotations)

        raw = raw.resample(tgt_fs)

        f_name = f.split('.')[0] + f'_S{j - start_idx + 1}' + '.set'
        mne.export.export_raw(os.path.join(out_path, f_name), raw, overwrite=True)
