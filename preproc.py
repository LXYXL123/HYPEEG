import logging
from typing import Optional, Type

from omegaconf import DictConfig, OmegaConf

from common.conf import BasePreprocArgs
from common.log import setup_log
from common.path import get_conf_file_path
from data.processor.builder import EEGDatasetBuilder
from data.processor.wrapper import DATASET_SELECTOR


logger = logging.getLogger('preproc')


def prepare_dataset(
        conf: BasePreprocArgs,
        builder_cls: Type[EEGDatasetBuilder],
        dataset_name: str,
        config_name: str,
        output_root: Optional[str] = None,
):
    try:
        logger.info(f"Preparing dataset {dataset_name} {config_name} at fs={conf.fs}Hz...")
        if output_root:
            logger.info(f"Using custom processed output root: {output_root}")
        builder = builder_cls(config_name, fs=conf.fs, database_proc_root=output_root)
        if conf.clean_middle_cache:
            builder.clean_disk_cache(clean_shared_info=conf.clean_shared_info)
        builder.preproc(n_proc=conf.num_preproc_mid_workers)
        # logger.info(f"Dataset {dataset_name} {config_name} is preprocessed.")
        builder.download_and_prepare(num_proc=conf.num_preproc_arrow_writers)
        dataset = builder.as_dataset()
        logger.info(f"Dataset {dataset_name} {config_name} at fs={conf.fs}Hz is prepared.")
        logger.info(f"{dataset}")
    except Exception as e:
        logger.error(f"Preparation of dataset {dataset_name} {config_name} exit with error: {e}.")


def preproc(conf: BasePreprocArgs):
    dataset_items: list[tuple[str, str, Optional[str]]] = []
    dataset_items.extend((dataset_name, 'pretrain', conf.output_root) for dataset_name in conf.pretrain_datasets)
    dataset_items.extend((dataset_name, config, conf.output_root) for dataset_name, config in conf.pretrain_dataset_configs.items())
    dataset_items.extend((dataset_name, config, None) for dataset_name, config in conf.finetune_datasets.items())

    for dataset, config, output_root in dataset_items:
        if dataset not in DATASET_SELECTOR.keys():
            raise ValueError(f"Dataset {dataset} is not supported.")

        builder_cls = DATASET_SELECTOR[dataset]
        if config not in builder_cls.builder_configs.keys():
            raise ValueError(f"Config {config} is not supported for dataset {dataset}.")

        prepare_dataset(conf, builder_cls, dataset, config, output_root=output_root)


if __name__ == '__main__':
    # cli args
    # conf_file = [abs path | rel path | file name]
    cli_args: DictConfig = OmegaConf.from_cli()
    logger.info(cli_args)
    if 'conf_file' in cli_args.keys():
        logger.info(cli_args.conf_file)
        file_cfg = OmegaConf.load(get_conf_file_path(cli_args.conf_file))
        cli_args.pop("conf_file")
    else:
        file_cfg = OmegaConf.create({})

    code_cfg = OmegaConf.create(BasePreprocArgs().model_dump())
    
    setup_log()
    cfg = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.info(cfg)
    cfg = BasePreprocArgs.model_validate(cfg)

    preproc(cfg)
