"""Synthetic full-bag throughput probe: actual architecture, no image preprocessing."""
import argparse
import gc
import json
import time
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime_fusion'))
import timm
import torch
from models.raptor_v5_radimagenet import COATNET_ARCH,FusedCoAtNet,FusionConfig,RadImageNetEncoder,RaptorClassifier
from scripts.train_fusion_ready import FullBagClassifier


def main(chunk,studies,repeats):
    torch.set_num_threads(4); torch.manual_seed(9)
    coat=timm.create_model(COATNET_ARCH,pretrained=False,num_classes=0,global_pool='avg')
    model=RaptorClassifier(FusedCoAtNet(coat,RadImageNetEncoder(),FusionConfig()),1024).cuda().train()
    network=FullBagClassifier(model,chunk,torch.float16)
    bags=[torch.randn(62,3,384,384,device='cuda') for _ in range(studies)]
    targets=torch.full((studies,12),.25,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    times=[]
    for step in range(repeats+1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        logits=network(bags)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite synthetic gradients')
        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
        seconds=time.monotonic()-started
        row=dict(device=torch.cuda.get_device_name(),chunk_size=chunk,studies_per_gpu=studies,
                 triplets_per_study=62,step=step,seconds=seconds,triplets_per_second=62*studies/seconds,
                 peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(loss.detach()))
        print(json.dumps(row),flush=True)
        if step:times.append(row)
        del logits,loss
    directory=ROOT/'results/fusion_training_hybrid50/benchmarks'
    directory.mkdir(exist_ok=True)
    (directory/f'chunk_{chunk}_studies_{studies}.json').write_text(json.dumps(times,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--chunk',type=int,required=True)
    parser.add_argument('--studies',type=int,default=2); parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    try:main(args.chunk,args.studies,args.repeats)
    except torch.OutOfMemoryError:
        print(json.dumps(dict(event='out_of_memory',chunk_size=args.chunk,studies_per_gpu=args.studies)),flush=True)
        sys.exit(2)
