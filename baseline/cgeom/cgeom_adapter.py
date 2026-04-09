"""
CGeom adapter for downstream EEG classification.
"""

import logging
from typing import List

from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin


logger = logging.getLogger("baseline")


class CGeomDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """Dataset adapter for the CGeom encoder."""

    def _setup_adapter(self):
        self.model_name = "cgeom"
        self.scale = 0.001  # convert uV to mV
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return self.get_standard_eeg_channels()


class CGeomDataLoaderFactory(AbstractDataLoaderFactory):
    """CGeom DataLoader factory."""

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> CGeomDatasetAdapter:
        return CGeomDatasetAdapter(dataset, dataset_names, dataset_configs)
