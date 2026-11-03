"""CPU checks for the DDP helpers (sharding, host EMA, gathered evaluation) using the gloo backend."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

import train_fusion_ddp as D  # noqa: E402
import train_fusion_v2 as T  # noqa: E402


class Sharding(unittest.TestCase):
    def test_disjoint_equal_shards(self):
        order = list(torch.randperm(3900).tolist())
        shards = []
        for rank in range(2):
            D.RANK, D.WORLD = rank, 2
            shards.append(D.rank_shard(order))
        D.RANK, D.WORLD = 0, 1
        self.assertEqual(len(shards[0]), len(shards[1]))
        self.assertEqual(len(shards[0]) + len(shards[1]), 3900)
        self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_odd_length_drops_one(self):
        D.RANK, D.WORLD = 1, 2
        self.assertEqual(len(D.rank_shard(list(range(7)))), 3)
        D.RANK, D.WORLD = 0, 1


class HostEMAMatchesDeviceEMA(unittest.TestCase):
    def test_same_numbers(self):
        torch.manual_seed(0)
        net = torch.nn.Linear(4, 3)
        a, b = T.EMA(net, 0.99), D.HostEMA(net, 0.99)
        for step in range(1, 30):
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p)*0.1)
            a.update(net, step); b.update(net, step)
        for x, y in zip(a.params, b.params):
            self.assertTrue(torch.allclose(x, y))
        trained = [p.detach().clone() for p in net.parameters()]
        backup = b.swap_in(net)
        for p, e in zip(net.parameters(), b.params):
            self.assertTrue(torch.equal(p, e))
        b.restore(net, backup)
        for p, t in zip(net.parameters(), trained):
            self.assertTrue(torch.equal(p, t))


def _ddp_worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.manual_seed(rank)   # deliberately different init per rank
    net = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    ddp = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False, gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    ema = D.HostEMA(net, 0.9)
    D.RANK, D.WORLD = rank, world
    data = [torch.randn(1, 6, generator=torch.Generator().manual_seed(100*rank+i)) for i in range(12)]
    accumulate = 4
    for i, x in enumerate(data):
        boundary = (i+1) % accumulate == 0
        ctx = ddp.no_sync() if not boundary else torch.enable_grad()
        with ctx:
            (ddp(x).square().mean()/accumulate).backward()
        if boundary:
            opt.step(); opt.zero_grad(set_to_none=True); ema.update(net, i)
    flat = torch.cat([p.detach().flatten() for p in net.parameters()])
    eflat = torch.cat([p.flatten() for p in ema.params])
    gathered = [None]*world
    dist.all_gather_object(gathered, (flat.tolist(), eflat.tolist()))
    if rank == 0:
        torch.save(gathered, out)
    dist.destroy_process_group()


class DDPLockstep(unittest.TestCase):
    def test_no_sync_accumulation_keeps_ranks_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'g.pt')
            mp.spawn(_ddp_worker, args=(2, 29561, out), nprocs=2, join=True)
            gathered = torch.load(out)
        (p0, e0), (p1, e1) = gathered
        self.assertTrue(torch.allclose(torch.tensor(p0), torch.tensor(p1)))
        self.assertTrue(torch.allclose(torch.tensor(e0), torch.tensor(e1)))


if __name__ == '__main__':
    unittest.main()
