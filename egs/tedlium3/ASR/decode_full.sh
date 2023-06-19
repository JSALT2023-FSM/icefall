# This script perform long-form decoding. It is based on https://github.com/k2-fsa/icefall/pull/980/.
#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export KALDI_ROOT=/home/hltcoe/draj/kaldi

set -eou pipefail

# This script is used to recogize long audios. The process is as follows:
# 1) Split long audios into chunks with overlaps.
# 2) Perform speech recognition on chunks, getting tokens and timestamps.
# 3) Merge the overlapped chunks into utterances acording to the timestamps.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
chunk=30
extra=4

stage=1
stop_stage=4
decoding_method=greedy_search

# scoring options
overlap_spk=2
hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

decode_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"
score_cmd="queue.pl --mem 12G"
. shared/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # Chunk manifests are saved to data/manifests/cuts_{subset}_chunks.jsonl.gz
  log "Stage 1: Split long audio into chunks"
  python local/split_into_chunks.py \
    --manifest-dir data/manifests \
    --chunk $chunk \
    --extra $extra  # Extra duration (in seconds) at both sides
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Perform speech recognition on splitted chunks"
  $decode_cmd zipformer/exp/v5/decode_chunked_greedy.log \
    python zipformer/decode_chunked.py \
      --epoch 99 --avg 1 \
      --exp-dir zipformer/exp/v5 \
      --manifest-dir data/manifests \
      --max-duration 600 \
      --decoding-method ${decoding_method} \
      --chunk $chunk --extra $extra \
      --beam-size 4
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Merge splitted chunks into utterances and score."
  python local/merge_chunks.py \
    --res-dir zipformer/exp/v5/${decoding_method}-chunked \
    --manifest-dir data/manifests \
    --bpe-model data/lang_bpe_500/bpe.model \
    --chunk $chunk --extra $extra
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute Asclite WER"
  for part in dev; do
    scoring_dir=zipformer/exp/v5/${decoding_method}-chunked/${part}_chunk${chunk}_extra${extra}_scoring
    $score_cmd $scoring_dir/scoring.log \
      $hubscr -G -v -m 1:2 -o$overlap_spk -a -C -B 8192 -p $hubdir \
        -V -l english -h rt-stt -g conf/dummy.glm \
        -r $scoring_dir/ref.stm $scoring_dir/hyp.ctm
  done
fi
