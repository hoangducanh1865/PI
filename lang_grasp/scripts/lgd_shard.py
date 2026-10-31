"""
Run the official LGD evaluate_diffusion.py on one shard of its test set, without changing its code.

Usage (from the LGD repo root, with its Python env; one process per shard, all shards at once):
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=<PORT> --use_env <THIS_FILE> <K> <N> <ARGS...>
<ARGS> are evaluate_diffusion.py's own arguments. The official script evaluates one sample at a time and spends most
of each diffusion step waiting for GPU-to-CPU copies (_process_attention_mask in inference/models/lgdm/network.py),
about 74 s per sample on an A100, so the test set is split over N processes: the sampler it builds keeps only the
indices i with i % N == K. Everything else (split, seed, sampler, metric) is the script's own; its log line
"IOU Results: c/n" is summed over the shards. The diffusion sampling is random, so a sharded run draws different
noise than one process would (the seed is set the same way in every shard). Each sample's result is also printed as
"LGD SAMPLE <0|1>" as soon as it is scored (by wrapping evaluation.calculate_iou_match), so a run stopped early still
gives a random subset of the test set (the script's sampler visits it in random order).

Resuming: with LGD_SKIP=<N_OLD>:<M> the samples already scored by an earlier run with N_OLD shards that stopped after
M samples per shard are left out. The script seeds torch (--random-seed 123) and its sampler draws its order with
torch.randperm from the global generator, so the earlier order is recomputed from the generator state at the start of
the iteration (the same state in every run) and the generator is restored before drawing the new order. Each sample's
dataset index is printed as "LGD INDEX <i>".

Author: Duc Anh Hoang
"""

import os
import runpy
import sys

import torch
import torch.utils.data.sampler as samplers

K, N = int(sys.argv[1]), int(sys.argv[2])
_Random = samplers.SubsetRandomSampler


SKIP = os.environ.get("LGD_SKIP")


class ShardSampler(_Random):
    def __init__(self, indices, generator=None):
        self.all_indices = list(indices)
        super().__init__([i for i in self.all_indices if i % N == K], generator)

    def __iter__(self):
        if SKIP:
            n_old, m = map(int, SKIP.split(":"))
            state = torch.get_rng_state()
            done = set()
            for k in range(n_old):
                old = [i for i in self.all_indices if i % n_old == k]
                torch.set_rng_state(state)
                done.update(old[j] for j in torch.randperm(len(old))[:m].tolist())
            torch.set_rng_state(state)
            self.indices = [i for i in self.all_indices if i % N == K and i not in done]
            rng = hash(state.numpy().tobytes())  # the same in every shard if the start state is reproducible
            print(f"LGD SKIP {len(done)} done, {len(self.indices)} left in shard {K}, rng {rng}", flush=True)
        for i in super().__iter__():
            print(f"LGD INDEX {i}", flush=True)
            yield i


samplers.SubsetRandomSampler = ShardSampler
sys.path.insert(0, os.getcwd())  # the LGD repo root
from utils.dataset_processing import evaluation  # noqa: E402  (the LGD repo's module)

_match = evaluation.calculate_iou_match


def logged_match(*args, **kwargs):
    hit = _match(*args, **kwargs)
    print(f"LGD SAMPLE {int(bool(hit))}", flush=True)
    return hit


evaluation.calculate_iou_match = logged_match
sys.argv = ["evaluate_diffusion.py", *sys.argv[3:]]
runpy.run_module("evaluate_diffusion", run_name="__main__")
