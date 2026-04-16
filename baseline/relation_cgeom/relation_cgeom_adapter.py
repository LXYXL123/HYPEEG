"""
Relation-CGeom adapter for downstream EEG classification.
"""

import logging
from typing import List

from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin
from common.utils import ElectrodeSet


logger = logging.getLogger("baseline")


class RelationCGeomDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """Dataset adapter for the relation-CGeom encoder."""

    @staticmethod
    def get_relation_cgeom_channels() -> List[str]:
        """Use the full 10-10 electrode vocabulary for setup-conditioned models."""
        return list(ElectrodeSet.Electrodes)

    def _setup_adapter(self):
        self.model_name = "relation_cgeom"
        self.scale = 0.001  # convert uV to mV
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return self.get_relation_cgeom_channels()


class RelationCGeomDataLoaderFactory(AbstractDataLoaderFactory):
    """Relation-CGeom dataloader factory."""

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> RelationCGeomDatasetAdapter:
        return RelationCGeomDatasetAdapter(dataset, dataset_names, dataset_configs)
