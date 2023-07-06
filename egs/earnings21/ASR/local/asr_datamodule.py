# Copyright      2021  Piotr Żelasko
# Copyright      2021  Xiaomi Corporation (Author: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (
    OnTheFlyFeatures,
    BatchIO,
    PrecomputedFeatures,
)
from torch.utils.data import DataLoader

from icefall.utils import str2bool

# I am an elite hacker
TedLiumAsrDataModule = None
LibriSpeechAsrDataModule = None

class SpeechRecognitionDataset(K2SpeechRecognitionDataset):
    def __init__(
        self,
        return_cuts: bool = False,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        super().__init__(return_cuts=return_cuts, input_strategy=input_strategy)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[Cut]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        batch = {"inputs": inputs, "supervisions": supervision_intervals}
        if self.return_cuts:
            batch["supervisions"]["cut"] = [cut for cut in cuts]

        return batch


class Earnings21AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. TEDLium3 dev
    and test).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-in",
            type=Path,
            help="Path to manifest with test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset.",
        )

    def train_dataloaders(
        self, cuts_train: CutSet, sampler_state_dict: Optional[Dict[str, Any]] = None
    ) -> DataLoader:
        raise NotImplementedError

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        raise NotImplementedError

    def test_dataloaders(self, cuts_test: CutSet, chunked: bool = False) -> DataLoader:
        logging.debug("About to create test dataset")
        if chunked:
            test = SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                if self.args.on_the_fly_feats
                else PrecomputedFeatures(),
                return_cuts=self.args.return_cuts,
            )
        else:
            test = K2SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                if self.args.on_the_fly_feats
                else PrecomputedFeatures(),
                return_cuts=self.args.return_cuts,
            )

        sampler = DynamicBucketingSampler(
            cuts_test,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("Nope")

    @lru_cache()
    def dev_cuts(self, affix: str = "") -> CutSet:
        logging.error("Nope")

    @lru_cache()
    def test_cuts(self, affix: str = "") -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(self.args.manifest_in)
