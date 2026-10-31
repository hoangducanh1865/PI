"""
Run the official LGD evaluate_diffusion.py on one shard of its test set, without changing its code.

Usage (from the LGD repo root, with its Python env; one process per shard, all shards at once):
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=<PORT> --use_env <THIS_FILE> <K> <N> <ARGS...>
<ARGS> are evaluate_diffusion.py's own arguments. The official script evaluates one sample at a time and spends most
of each diffusion step waiting for GPU-to-CPU copies (_process_attention_mask in inference/models/lgdm/network.py),
about 74 s per sample on an A100, so the test set is split over N processes: the sampler it builds keeps only the
indices i with i % N == K. Everything else (split, seed, sampler, metric) is the script's own; its log line
"IOU Results: c/n" is summed over the shards. The diffusion sampling is random, so a sharded run draws different
noise than one process would (the seed is set the same way in every shard).

Author: Duc Anh Hoang
"""

import os
import runpy
import sys

import torch.utils.data.sampler as samplers

K, N = int(sys.argv[1]), int(sys.argv[2])
_Random = samplers.SubsetRandomSampler


class ShardSampler(_Random):
    def __init__(self, indices, generator=None):
        super().__init__([i for i in indices if i % N == K], generator)


samplers.SubsetRandomSampler = ShardSampler
sys.path.insert(0, os.getcwd())  # the LGD repo root
sys.argv = ["evaluate_diffusion.py", *sys.argv[3:]]
runpy.run_module("evaluate_diffusion", run_name="__main__")
