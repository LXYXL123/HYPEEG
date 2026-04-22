import logging
import re
from dataclasses import dataclass
from typing import Type, Optional

import torch
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets, Value

from data.dataset.adftd import AdftdBuilder
from data.dataset.bcic.bcic_1a import BCIC1ABuilder
from data.dataset.bcic.bcic_2020_3 import BCIC2020ImagineBuilder
from data.dataset.bcic.bcic_2a import BCIC2ABuilder
from data.dataset.brain_lat import BrainLatBuilder
from data.dataset.chisco import ChiscoBuilder
from data.dataset.cho2017 import Cho2017Builder
from data.dataset.emobrain import EmobrainBuilder
from data.dataset.grasp_and_lift import GraspAndLiftBuilder
from data.dataset.hbn import HBNBuilder
from data.dataset.hmc import HMCBuilder
from data.dataset.inner_speech import InnerSpeechBuilder
from data.dataset.inria_bci import InriaBciBuilder
from data.dataset.mimul_11 import Mimul11Builder
from data.dataset.motor_mv_img import MotorMoveImagineBuilder
from data.dataset.openmiir import OpenMiirBuilder
from data.dataset.seeds.seed import SeedBuilder
from data.dataset.seeds.seed_fra import SeedFraBuilder
from data.dataset.seeds.seed_ger import SeedGerBuilder
from data.dataset.seeds.seed_iv import SeedIVBuilder
from data.dataset.seeds.seed_v import SeedVBuilder
from data.dataset.seeds.seed_vii import SeedVIIBuilder
from data.dataset.siena_scalp import SienaScalpBuilder
from data.dataset.spis_resting_state import SpisRestingStateBuilder
from data.dataset.target_versus_non import TargetVersusNonBuilder
from data.dataset.things_eeg import ThingsEEGBuilder
from data.dataset.things_eeg_2 import ThingsEEG2Builder
from data.dataset.trujillo_2017 import Trujillo2017Builder
from data.dataset.trujillo_2019 import Trujillo2019Builder
from data.dataset.tue.tuab import TuabBuilder
from data.dataset.tue.tuar import TuarBuilder
from data.dataset.tue.tueg import TuegBuilder
from data.dataset.tue.tuep import TuepBuilder
from data.dataset.tue.tuev import TuevBuilder
from data.dataset.tue.tusl import TuslBuilder
from data.dataset.tue.tusz import TuszBuilder
from data.dataset.workload import WorkloadBuilder
from data.processor.builder import EEGDatasetBuilder, EEGConfig


log = logging.getLogger()


def _patch_datasets_local_filesystem_detection() -> None:
    """Keep older datasets compatible with newer fsspec LocalFileSystem.

    Some datasets releases check ``fs.protocol != "file"`` to decide whether a
    cache is remote. Newer fsspec may expose LocalFileSystem.protocol as a tuple
    like ("file", "local"), which makes datasets wrongly reject local caches.
    """
    try:
        import datasets.builder as datasets_builder
        import datasets.filesystems as datasets_filesystems
    except Exception:
        return

    def is_remote_filesystem(fs) -> bool:
        protocol = getattr(fs, 'protocol', None)
        if isinstance(protocol, (tuple, list, set)):
            return 'file' not in protocol and 'local' not in protocol
        return protocol not in ('file', 'local', None)

    datasets_filesystems.is_remote_filesystem = is_remote_filesystem
    datasets_builder.is_remote_filesystem = is_remote_filesystem


_patch_datasets_local_filesystem_detection()


@dataclass(frozen=True)
class RuntimeDatasetConfig:
    base_config: str
    split_mode: Optional[str] = None
    fold_index: Optional[int] = None


BCIC2A_LOSO_SPLITS = [
    {"test": [1], "val": [2, 3], "train": [4, 5, 6, 7, 8, 9]},
    {"test": [2], "val": [3, 4], "train": [1, 5, 6, 7, 8, 9]},
    {"test": [3], "val": [4, 5], "train": [1, 2, 6, 7, 8, 9]},
    {"test": [4], "val": [5, 6], "train": [1, 2, 3, 7, 8, 9]},
    {"test": [5], "val": [6, 7], "train": [1, 2, 3, 4, 8, 9]},
    {"test": [6], "val": [7, 8], "train": [1, 2, 3, 4, 5, 9]},
    {"test": [7], "val": [8, 9], "train": [1, 2, 3, 4, 5, 6]},
    {"test": [8], "val": [9, 1], "train": [2, 3, 4, 5, 6, 7]},
    {"test": [9], "val": [1, 2], "train": [3, 4, 5, 6, 7, 8]},
]

BCIC2A_LOSO_CONFIG_PATTERN = re.compile(r"^(?P<base>finetune)_loso(?:_fold)?(?P<fold>\d+)$")


DATASET_SELECTOR: dict[str, Type[EEGDatasetBuilder]] = {
    'tuab': TuabBuilder,
    'tuar': TuarBuilder,
    'tueg': TuegBuilder,
    'tuep': TuepBuilder,
    'tuev': TuevBuilder,
    'tusl': TuslBuilder,
    'tusz': TuszBuilder,
    'seed': SeedBuilder,
    'seed_fra': SeedFraBuilder,
    'seed_ger': SeedGerBuilder,
    'seed_iv': SeedIVBuilder,
    'seed_v': SeedVBuilder,
    'seed_vii': SeedVIIBuilder,
    'bcic_1a': BCIC1ABuilder,
    'bcic_2a': BCIC2ABuilder,
    'bcic_2020_3': BCIC2020ImagineBuilder,
    'emobrain': EmobrainBuilder,
    'grasp_and_lift': GraspAndLiftBuilder,
    'hmc': HMCBuilder,
    'inria_bci': InriaBciBuilder,
    'motor_mv_img': MotorMoveImagineBuilder,
    'siena_scalp': SienaScalpBuilder,
    'spis_resting_state': SpisRestingStateBuilder,
    'target_versus_non': TargetVersusNonBuilder,
    'trujillo_2017': Trujillo2017Builder,
    'trujillo_2019': Trujillo2019Builder,
    'workload': WorkloadBuilder,
    'hbn': HBNBuilder,
    'adftd': AdftdBuilder,
    'brain_lat': BrainLatBuilder,
    'things_eeg': ThingsEEGBuilder,
    'things_eeg_2': ThingsEEG2Builder,
    'mimul_11': Mimul11Builder,
    'inner_speech': InnerSpeechBuilder,
    'chisco': ChiscoBuilder,
    'cho2017': Cho2017Builder,
    'open_miir': OpenMiirBuilder,
}


def resolve_runtime_dataset_config(dataset_name: str, config_name: str) -> RuntimeDatasetConfig:
    if dataset_name == 'bcic_2a':
        match = BCIC2A_LOSO_CONFIG_PATTERN.fullmatch(config_name)
        if match is not None:
            fold_number = int(match.group('fold'))
            if fold_number < 1 or fold_number > len(BCIC2A_LOSO_SPLITS):
                raise ValueError(
                    f'Invalid BCIC-2a LOSO fold {fold_number}. '
                    f'Expected 1..{len(BCIC2A_LOSO_SPLITS)}.'
                )
            return RuntimeDatasetConfig(
                base_config=match.group('base'),
                split_mode='loso',
                fold_index=fold_number - 1,
            )
    return RuntimeDatasetConfig(base_config=config_name)


def _map_runtime_split_name(split: datasets.NamedSplit) -> str:
    if split == datasets.Split.TRAIN:
        return 'train'
    if split == datasets.Split.VALIDATION:
        return 'val'
    if split == datasets.Split.TEST:
        return 'test'
    raise ValueError(f'Unsupported split for runtime dataset routing: {split}')


def _load_bcic2a_loso_dataset(
        builder: EEGDatasetBuilder,
        split: datasets.NamedSplit,
        fold_index: int,
) -> Dataset:
    target_split = _map_runtime_split_name(split)
    target_subjects = {str(subj) for subj in BCIC2A_LOSO_SPLITS[fold_index][target_split]}

    source_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
    full_dataset = concatenate_datasets([builder.as_dataset(split=source_split) for source_split in source_splits])
    full_count = len(full_dataset)
    filtered = full_dataset.filter(
        lambda subject: str(subject) in target_subjects,
        input_columns=['subject'],
        desc=f'Filtering bcic_2a LOSO fold {fold_index + 1} {target_split}',
    )
    log.info(
        f'Loading bcic_2a LOSO fold {fold_index + 1}: '
        f'split={target_split}, subjects={sorted(int(s) for s in target_subjects)}, '
        f'samples={len(filtered)}/{full_count}'
    )
    return filtered


def load_runtime_eeg_dataset(
        dataset_name: str,
        builder_config: str,
        split: datasets.NamedSplit,
        fs: int,
) -> Dataset:
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, builder_config)
    builder_cls = DATASET_SELECTOR[dataset_name]
    builder = builder_cls(config_name=runtime_cfg.base_config, fs=fs)

    if runtime_cfg.split_mode == 'loso':
        if dataset_name != 'bcic_2a' or runtime_cfg.fold_index is None:
            raise ValueError(f'Unsupported runtime split mode for {dataset_name}-{builder_config}')
        return _load_bcic2a_loso_dataset(builder, split=split, fold_index=runtime_cfg.fold_index)

    # noinspection PyTypeChecker
    return builder.as_dataset(split=split)

def get_dataset_patch_len(dataset_name: str, config_name: str) -> int:
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, config_name)
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(runtime_cfg.base_config)
    return config.wnd_div_sec


def get_dataset_shape_info(dataset_name: str, config_name: str, fs: int) -> dict[str, tuple[int, int]]:
    """
    Get shape information for each montage in a dataset.

    Args:
        dataset_name: Name of the dataset
        config_name: Configuration name
        fs: Sampling frequency

    Returns:
        Dict mapping montage_key -> (n_timepoints, n_channels)
    """
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, config_name)
    builder_cls = DATASET_SELECTOR[dataset_name]
    builder: EEGDatasetBuilder = builder_cls(config_name=runtime_cfg.base_config)

    config: EEGConfig = builder.config
    n_timepoints = int(config.wnd_div_sec * fs)

    shape_info: dict[str, tuple[int, int]] = {}
    for montage_name in config.montage.keys():
        montage_key = f'{dataset_name}/{montage_name}'
        chs = builder.standardize_chs_names(montage_name)
        n_channels = len(chs)
        shape_info[montage_key] = (n_timepoints, n_channels)

    return shape_info


def get_dataset_n_class(dataset_name: str, config_name: str) -> int:
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, config_name)
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(runtime_cfg.base_config)
    return len(config.category)

def get_dataset_category(dataset_name: str, config_name: str) -> list[str]:
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, config_name)
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(runtime_cfg.base_config)
    return config.category

def get_dataset_montage(dataset_name: str, config_name: str) -> dict[str, list[str]]:
    # Note: This function needs builder instance to call standardize_chs_names()
    runtime_cfg = resolve_runtime_dataset_config(dataset_name, config_name)
    builder_cls = DATASET_SELECTOR[dataset_name]
    builder: EEGDatasetBuilder = builder_cls(config_name=runtime_cfg.base_config)
    montage_names = builder.config.montage.keys()

    montages: dict[str, list[str]] = dict()
    for montage_name in montage_names:
        montages[f'{dataset_name}/{montage_name}'] = builder.standardize_chs_names(montage_name)

    return montages


def load_concat_eeg_datasets(
        dataset_names: list[str],
        builder_configs: list[str],
        split: datasets.NamedSplit = datasets.Split.TRAIN,
        weight_option: str = 'statistics',
        add_ds_name: bool = False,
        cast_label: bool = False,
        fs: Optional[int] = None,
) -> tuple[Dataset, list[Tensor]]:
    """
    Load and concatenate multiple EEG datasets.

    :param dataset_names: List of dataset names to load
    :param builder_configs: List of builder config names (e.g., 'pretrain', 'finetune')
    :param split: Dataset split to load (TRAIN, VALIDATION, TEST)
    :param weight_option: Weight calculation option for class imbalance
    :param add_ds_name: Whether to add dataset name column
    :param cast_label: Whether to cast label to int64
    :param fs: Target sampling rate (must match preprocessed data)
    :return: Tuple of concatenated dataset and weight list
    """
    dataset_list = []
    weight_list = []

    if fs is None:
        raise ValueError('fs for dataset loader must be specified')

    for ds_name, ds_config in zip(dataset_names, builder_configs):
        try:
            runtime_cfg = resolve_runtime_dataset_config(ds_name, ds_config)
            builder_cls = DATASET_SELECTOR[ds_name]
            builder = builder_cls(config_name=runtime_cfg.base_config, fs=fs)
            log.info(f'Loading {ds_name}-{ds_config} at fs={fs}Hz from {builder.cache_dir}')
            dataset = load_runtime_eeg_dataset(
                dataset_name=ds_name,
                builder_config=ds_config,
                split=split,
                fs=fs,
            )
            if add_ds_name:
                dataset = dataset.add_column('ds_name', [ds_name for _ in range(len(dataset))])

            if 'label' in dataset.column_names:
                n_class = get_dataset_n_class(ds_name, ds_config)
                if n_class > 1:
                    label = torch.tensor(dataset['label'], dtype=torch.int32)
                    label_cnt = torch.bincount(label, minlength=n_class)
                    log.info(f'Sample distribution for {ds_name}-{ds_config} {split}: {label_cnt}')
                    weight = calc_distribution_weight(len(dataset), label_cnt, weight_option)
                    weight_list.append(weight)

                    if cast_label:
                        dataset = dataset.cast_column('label', Value('int64'))
                else:
                    # Regression: do NOT cast label to int64; and use a dummy weight tensor for API consistency.
                    log.info(
                        f'Sample distribution for {ds_name}-{ds_config} {split}: regression (n_class=1), skip bincount'
                    )
                    weight_list.append(torch.ones(1, dtype=torch.int64))

            dataset_list.append(dataset)
        except KeyError:
            log.error(f'Dataset {ds_name} not found')

    combined_dataset: Dataset = concatenate_datasets(dataset_list)
    # combined_dataset = combined_dataset.flatten_indices()
    return combined_dataset.with_format('torch'), weight_list

def calc_distribution_weight(n: int, label_cnt: Tensor, option: str):
    if option == 'statistics':
        return label_cnt
    elif option == 'sqrt':
        return n / torch.sqrt(label_cnt.float() + 1)
    elif option == 'log':
        return n / torch.log(label_cnt.float() + 1)
    elif option == 'absolute':
        return n / label_cnt.float()
    else:
        raise ValueError(f'Unknown option {option}')



if __name__ == '__main__':
    # data = load_concat_eeg_datasets(['seed_v', 'tuab'])
    data, distribution = load_concat_eeg_datasets(['tuab'], ['finetune'], fs=256)
    loader = DataLoader(data, batch_size=32)

    for batch in loader:
        pass
