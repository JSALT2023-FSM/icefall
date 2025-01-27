#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script splits long utterances into chunks with overlaps.
Each chunk (except the first and the last) is padded with extra left side and right side.
The chunk length is: left_side + chunk_size + right_side.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, SupervisionSet, RecordingSet


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
        "--manifest-out",
        type=Path,
        help="Output path to cut manifest.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))

    recording_set = RecordingSet.from_jsonl(args.recording_manifest)
    supervision_set = SupervisionSet.from_jsonl(args.supervision_manifest)

    cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )

    cuts.to_file(args.manifest_out)
    logging.info(f"Cuts saved to {args.manifest_out}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()

