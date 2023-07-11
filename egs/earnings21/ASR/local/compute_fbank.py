#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# 	       2022  Xiaomi Crop.        (authors: Mingshuang Luo)
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
"""
This file computes fbank features of the TedLium3 dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, SupervisionSet, RecordingSet, MultiCut
from lhotse.audio import set_audio_duration_mismatch_tolerance
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

set_audio_duration_mismatch_tolerance(30.0)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recording-manifest",
        type=Path,
        help="Path to recording manifest.",
    )
    parser.add_argument(
        "--supervision-manifest",
        type=Path,
        help="Path to supervision manifest.",
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Path to cut manifest.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Output path to cut manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for features.",
    )
    return parser.parse_args()


def compute_fbank(args):
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        if not args.manifest_in:
            recording_set = RecordingSet.from_jsonl(args.recording_manifest)
            supervision_set = SupervisionSet.from_jsonl(args.supervision_manifest)

            cut_set = CutSet.from_manifests(
                recordings=recording_set,
                supervisions=supervision_set
            )
        else:
            cut_set = CutSet.from_file(args.manifest_in)

        cur_num_jobs = min(num_jobs, len(cut_set))

        cut_set = cut_set.resample(16000)
        cut_set = CutSet.from_cuts(c.to_mono(mono_downmix=True) if isinstance(c, MultiCut) else c for c in cut_set)
        cut_set = cut_set.save_audios(f"{args.output_dir}/wavs")
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{args.output_dir}/feats",
            # when an executor is specified, make more partitions
            num_jobs=cur_num_jobs,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        # Split long cuts into many short and un-overlapping cuts
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_file(args.manifest_out)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank(get_args())
