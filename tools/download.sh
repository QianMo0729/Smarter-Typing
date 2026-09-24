#!/bin/zsh
# Parallel downloads from hf-mirror.com (faster than huggingface.co from here).
set -u
M=https://hf-mirror.com
dl() { # url outdir outname
  aria2c -x 16 -s 16 -k 1M --file-allocation=none --console-log-level=warn --summary-interval=0 -c -d "$2" -o "$3" "$1"
}
cd "$(dirname $0)/.."
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do
  dl $M/Qwen/Qwen3-0.6B-Base/resolve/main/$f models/hf/Qwen3-0.6B-Base $f
  dl $M/Qwen/Qwen2.5-0.5B/resolve/main/$f models/hf/Qwen2.5-0.5B $f
done
dl $M/datasets/0xDing/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json data/raw wiki.json
dl $M/datasets/wangrui6/Zhihu-KOL/resolve/main/data/train-00000-of-00005-a1278ede4e8c5cdb.parquet data/raw zhihu0.parquet
dl $M/datasets/wangrui6/Zhihu-KOL/resolve/main/data/train-00001-of-00005-1fc2da944397e9c2.parquet data/raw zhihu1.parquet
dl $M/datasets/Dialogue-Model-Research-Group/v2ex/resolve/main/v2ex_topic.jsonl.gz data/raw v2ex_topic.jsonl.gz
dl $M/datasets/Dialogue-Model-Research-Group/v2ex/resolve/main/v2ex_replies.jsonl.gz data/raw v2ex_replies.jsonl.gz
dl $M/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl data/raw cci3_0.jsonl
echo ALLDONE
