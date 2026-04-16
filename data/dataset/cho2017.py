import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import datasets
import mne
import numpy as np
import pandas as pd
import s3fs
from mne.io import BaseRaw, RawArray
from pandas import DataFrame
from scipy.io import loadmat

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class Cho2017Config(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "Cho2017 motor imagery dataset: 52 subjects, 64 EEG channels, left-hand "
        "and right-hand motor imagery, originally sampled at 512Hz."
    )
    citation: Optional[str] = "https://doi.org/10.1093/gigascience/gix034"

    filter_low: float = 0.5
    filter_high: float = 45.0
    filter_notch: float = 60.0
    is_notched: bool = False

    dataset_name: Optional[str] = 'cho2017'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY
    file_ext: str = 'mat'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
            'FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
            'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
            'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
            'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ',
            'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4',
            'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ',
            'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
            'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
        ]
    })

    valid_ratio: float = 0.12
    test_ratio: float = 0.12
    wnd_div_sec: int = 3
    suffix_path: str = 'Cho2017'
    scan_sub_dir: str = 'mat_data'

    category: list[str] = field(default_factory=lambda: [
        'left_hand', 'right_hand'
    ])


class Cho2017Builder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = Cho2017Config
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='pretrain_bci'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, _, files in os.walk(scan_path):
            for file in files:
                if re.fullmatch(r's\d{2}\.mat', file.lower()):
                    raw_data_files.append(os.path.normpath(os.path.join(root, file)))
        return sorted(raw_data_files)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        match = re.search(r's(\d+)', file_name.lower())
        if match is None:
            raise ValueError(f'Invalid Cho2017 file name: {file_path}')
        return {
            'subject': int(match.group(1)),
            'session': 1,
        }

    def _load_eeg_struct(self, file_path: str) -> dict[str, Any]:
        mat = loadmat(file_path, simplify_cells=True)
        if 'eeg' not in mat:
            raise KeyError(f'Cho2017 file has no eeg struct: {file_path}')
        return mat['eeg']

    @staticmethod
    def _event_onsets(event_signal: np.ndarray) -> np.ndarray:
        event_signal = np.asarray(event_signal).reshape(-1)
        active = event_signal > 0
        if active.size == 0:
            return np.array([], dtype=np.int64)
        prev = np.concatenate([[False], active[:-1]])
        return np.flatnonzero(active & ~prev).astype(np.int64)

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_10',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        raw_mat = self._load_eeg_struct(file_path)
        left = np.asarray(raw_mat['imagery_left'])
        right = np.asarray(raw_mat['imagery_right'])
        event_signal = np.asarray(raw_mat['imagery_event'])

        # The .mat stores left/right MI blocks separately. We concatenate them
        # in _read_raw_data with a short zero gap, so right-hand event onsets
        # need the same offset here.
        gap_samples = 500
        left_len = left.shape[1]
        right_offset = left_len + gap_samples

        annotations = []
        for onset in self._event_onsets(event_signal):
            annotations.append((
                'left_hand' if self.config.is_finetune else 'default',
                round(onset * 1000 / 512.0),
                round((onset + self.config.wnd_div_sec * 512) * 1000 / 512.0),
            ))
        for onset in self._event_onsets(event_signal):
            onset = onset + right_offset
            annotations.append((
                'right_hand' if self.config.is_finetune else 'default',
                round(onset * 1000 / 512.0),
                round((onset + self.config.wnd_div_sec * 512) * 1000 / 512.0),
            ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if self.config.is_finetune:
            df.loc[df['subject'].isin(np.arange(1, 37)), 'split'] = 'train'
            df.loc[df['subject'].isin(np.arange(37, 45)), 'split'] = 'valid'
            df.loc[df['subject'].isin(np.arange(45, 53)), 'split'] = 'test'
        else:
            df.loc[df['subject'].isin(np.arange(1, 47)), 'split'] = 'train'
            df.loc[df['subject'].isin(np.arange(47, 53)), 'split'] = 'valid'
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]
        if montage != '10_10':
            raise ValueError('No such montage in Cho2017 dataset')
        chs_std = [ch.upper() for ch in self.config.montage[montage]]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        raw_mat = self._load_eeg_struct(file_path)
        left = np.asarray(raw_mat['imagery_left'], dtype=np.float32)[:64] * 1e-6
        right = np.asarray(raw_mat['imagery_right'], dtype=np.float32)[:64] * 1e-6
        gap = np.zeros((left.shape[0], 500), dtype=np.float32)
        data = np.concatenate([left, gap, right], axis=1)

        info = mne.create_info(
            ch_names=self.config.montage['10_10'],
            sfreq=512.0,
            ch_types=['eeg'] * len(self.config.montage['10_10']),
        )
        raw = RawArray(data, info=info, verbose=verbose)
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})
            raw.set_montage(montage, on_missing='ignore', verbose=False)
        except Exception:
            logger.warning(f'Failed to set standard_1005 montage for {file_path}', exc_info=True)
        return raw

    def _persist_example_file(self, sample: dict):
        path, montage, label, split, subject = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'], sample['subject'])
        try:
            with self._read_raw_data(path, preload=True, verbose=False) as data:
                data = self._select_data_channels(data, path, montage)
                data = self._resample_and_filter(data)
                raw = self._fetch_signal_ndarray(data)
                chs_idx = self._fetch_chs_index(montage)

                examples = self._generate_window_sample(raw, montage, chs_idx, label, self.config.persist_drop_last)
                if len(examples) < 1:
                    return None

                df = pd.DataFrame(data=examples)
                df['subject'] = str(subject)
                filename = f"{self._encode_path(path)}.parquet"
                output_path = self._build_output_dir(split, filename)

                if self.config.is_remote_fs:
                    fs = s3fs.S3FileSystem(**self.s3_conf)
                    with fs.open(output_path, 'wb') as f:
                        df.to_parquet(
                            f,
                            compression=self.config.mid_compress_algo,
                            engine='pyarrow',
                            index=False)
                    fs.invalidate_cache()
                else:
                    df.to_parquet(
                        output_path,
                        compression=self.config.mid_compress_algo,
                        engine='pyarrow',
                        index=False)
        except Exception as e:
            logger.error(f"Error persisting example file {path}: {str(e)}")
            return None

        mid_df = pd.DataFrame(data={
            'key': [filename],
            'split': [split],
            'cnt': [len(examples)],})
        return mid_df


if __name__ == "__main__":
    builder = Cho2017Builder('pretrain_bci')
    builder.preproc(n_proc=2)
    builder.download_and_prepare(num_proc=2)
    dataset = builder.as_dataset()
    print(dataset)
