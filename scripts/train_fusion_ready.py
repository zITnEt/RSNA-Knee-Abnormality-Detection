"""Train the fusion branch on saved tensors. No pixel transformations occur."""
import json
import math
import random
import time
from pathlib import Path
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.checkpoint import checkpoint

from models.raptor_v5_radimagenet import LABELS, FusionConfig, build_raptor_v5_radimagenet, optimizer_groups, fused_checkpoint, load_fused_checkpoint
from raptor_data.ready_tensors import ReadyTripletDataset

RECIPE_SHA = '26ba1f6d2e41f37150239c8d46f738f2a0eca9e6b5a89a7542724c9f4c44aaca'
DEFAULTS = dict(epochs=50, k=None, seed=20261009, validation_fraction=0.2, chunk_size=8,
                studies_per_gpu=2, effective_batch=16, workers=2, backbone_lr=1e-5, fusion_lr=1e-4,
                head_lr=1e-4, weight_decay=0.01, max_hours=4.5,
                save_every_steps=25, resume_path=None)


def collate_studies(items):
    # Different studies have different K. Keep every triplet without padding.
    return dict(images=[item['images'] for item in items],
                targets=torch.stack([item['targets'] for item in items]),
                study_uid=[item['study_uid'] for item in items])


@contextmanager
def preserve_batchnorm_buffers(module):
    # Recomputation must not count the same chunk a second time in BN statistics.
    saved = [(buffer, buffer.clone()) for child in module.modules()
             if isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
             for buffer in child.buffers(recurse=False) if buffer is not None]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


class FullBagClassifier(torch.nn.Module):
    """Every chunk contributes to one full-bag attention head and its gradients."""
    def __init__(self, model, chunk_size, amp_dtype=None):
        super().__init__()
        self.model, self.chunk_size = model, chunk_size
        self.amp_dtype = amp_dtype

    def forward(self, bag):
        bags = [bag] if isinstance(bag, torch.Tensor) else bag
        if not bags or any(x.ndim != 4 or not len(x) for x in bags):
            raise ValueError('Expected one nonempty [K,3,H,W] study bag')
        lengths = [len(x) for x in bags]
        bag = torch.cat(bags, dim=0) if len(bags)>1 else bags[0]
        # Set dtype inside each worker thread; parallel_apply only copies AMP enablement.
        amp = torch.autocast(bag.device.type, dtype=self.amp_dtype) if self.amp_dtype else nullcontext()
        with amp:
            features = []
            for chunk in bag.split(self.chunk_size):
                if self.training and torch.is_grad_enabled():
                    features.append(checkpoint(self.model.backbone, chunk, use_reentrant=False,
                        context_fn=lambda:(nullcontext(), preserve_batchnorm_buffers(self.model.backbone))))
                else:
                    features.append(self.model.backbone(chunk))
            features = torch.cat(features, dim=0)
            return torch.cat([self.model.head(x[None]) for x in features.split(lengths)], dim=0)


class RaggedBagParallel(torch.nn.Module):
    """Several complete variable-length studies per GPU; gather study logits."""
    def __init__(self, module, devices, studies_per_gpu=1):
        super().__init__()
        self.module, self.devices = module, devices
        self.studies_per_gpu = studies_per_gpu

    def forward(self, bags):
        groups = [bags[i:i+self.studies_per_gpu] for i in range(0,len(bags),self.studies_per_gpu)]
        devices = self.devices[:len(groups)]
        if len(groups)>len(devices):
            raise ValueError('Study batch exceeds configured GPU capacity')
        inputs = [([bag.to(f'cuda:{device}', non_blocking=True) for bag in group],)
                  for group, device in zip(groups, devices)]
        if len(devices)==1:
            return self.module(*inputs[0])
        replicas = torch.nn.parallel.replicate(self.module, devices)
        outputs = torch.nn.parallel.parallel_apply(replicas, inputs, devices=devices)
        return torch.nn.parallel.gather(outputs, devices[0], dim=0)


def batch_settings(gpu_count, studies_per_gpu, effective_batch):
    devices = min(gpu_count,2)
    batch = devices*studies_per_gpu
    if batch < 1 or effective_batch % batch:
        raise ValueError('Effective batch must be divisible by the GPU study batch')
    return devices, batch, effective_batch//batch


def resume_batch_position(checkpoint_data, batch_size, studies):
    prior_batch_size = checkpoint_data.get('batch_size',2)
    cursor = checkpoint_data.get('next_study', min(checkpoint_data['next_batch']*prior_batch_size,studies))
    if not 0 <= cursor <= studies or (cursor!=studies and cursor%batch_size):
        raise ValueError('Checkpoint study position cannot be mapped to the new batch size')
    return math.ceil(cursor/batch_size)


def accumulation_weight(batch_index, batch_size, accumulate, total_studies, current_studies):
    start = (batch_index//accumulate)*accumulate*batch_size
    return current_studies/min(batch_size*accumulate,total_studies-start)


def load_hybrid_labels(path, required_ids):
    targets = pd.read_csv(path, dtype={'StudyInstanceUID':str})
    if targets.StudyInstanceUID.duplicated().any():
        raise ValueError('Duplicate HYBRID study IDs')
    targets = targets.set_index('StudyInstanceUID')
    if not set(required_ids) <= set(targets.index):
        raise ValueError('HYBRID targets do not cover the non-gold corpus')
    values = targets.loc[list(required_ids), list(LABELS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Invalid HYBRID soft targets in training/validation studies')
    return dict(zip(required_ids, values))


def event(kind, **data):
    record = dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data)
    print(json.dumps(record), flush=True)
    with Path('training_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    Path('progress.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def discover_roots():
    roots = []
    for home in [Path('/kaggle/input'), Path('/kaggle/input/datasets/zafarnematov')]:
        roots.extend(home.glob('rsna-six-slot-raptor-v5-shard-*'))
    for home in [Path('/kaggle/input'), Path('/kaggle/input/notebooks/zafarnematov')]:
        for notebook in home.glob('rsna-six-slot-raptor-v5-preprocessing-*'):
            roots.extend(notebook.glob('six_slot_raptor_shard_*'))
    roots = sorted(set(p for p in roots if (p/'recipe.json').is_file()))
    if len(roots) != 32:
        raise ValueError(f'Expected all 32 completed shards; found {len(roots)}')
    for root in roots:
        recipe = json.loads((root/'recipe.json').read_text())
        manifest = json.loads((root/'manifest.json').read_text())
        if recipe['sha256'] != RECIPE_SHA or not recipe['normalized'] or not manifest['complete']:
            raise ValueError(f'Incorrect/incomplete preprocessing shard: {root}')
    return roots


def asset(slug, filename):
    paths = sorted(Path('/kaggle/input').rglob(filename))
    paths = [p for p in paths if slug in str(p)]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one {slug}/{filename}; found {paths}')
    return paths[0]


def split_indices(files, seed, fraction):
    order = np.random.default_rng(seed).permutation(len(files))
    nval = round(len(files)*fraction)
    if not 0 < nval < len(files):
        raise ValueError('Validation fraction must leave nonempty training and validation sets')
    return sorted(order[nval:].tolist()), sorted(order[:nval].tolist())


def atomic_save(data, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, loader, dtype, chunk=12):
    # Full exported bags; encode in chunks for memory, then one attention MIL head.
    model.eval()
    total, count = 0., 0
    for batch in loader:
        images = batch['images'][0].cuda(non_blocking=True)
        if not torch.isfinite(images).all():
            raise ValueError('Non-finite saved validation tensor')
        with torch.autocast('cuda', dtype=dtype):
            features = torch.cat([model.backbone(x) for x in images.split(chunk)], dim=0)[None]
            logits = model.head(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), batch['targets'].cuda())
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite validation loss')
        total += float(loss); count += 1
        if count % 50 == 0:
            event('validation_progress', studies=count, total=len(loader), loss=total/count)
    model.train()
    return total/count


def main(config=None):
    started = time.monotonic()
    cfg = {**DEFAULTS, **(config or {})}
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requested; refusing CPU fallback')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    roots = discover_roots()
    if cfg['k'] is not None:
        raise ValueError('This run must use all available triplets: k=None')
    train_data = ReadyTripletDataset(roots, split='train', k=None)
    labels_path = asset('jev-knee-labels', 'HYBRID_labels_scores.csv')
    import hashlib
    with labels_path.open('rb') as stream:
        label_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    cfg['label_sha256'] = label_sha256
    labels = load_hybrid_labels(labels_path, [p.stem for p in train_data.files])
    train_data.labels = labels
    eval_data = ReadyTripletDataset(roots, split='train', k=None, labels=labels)
    gold_data = ReadyTripletDataset(roots, split='held_out_gold', k=None)
    if len(train_data) != 4333 or len(gold_data) != 74:
        raise ValueError(f'Corpus split mismatch: {len(train_data)}, {len(gold_data)}')
    train_idx, val_idx = split_indices(train_data.files, cfg['seed'], cfg['validation_fraction'])
    train_ids = [train_data.files[i].stem for i in train_idx]
    val_ids = [train_data.files[i].stem for i in val_idx]
    assert not (set(train_ids) | set(val_ids)) & {p.stem for p in gold_data.files}
    assert train_data.files == eval_data.files
    split = dict(train=train_ids, validation=val_ids, heldout_gold=[p.stem for p in gold_data.files])
    Path('training_split.json').write_text(json.dumps(split, indent=2))
    Path('training_config.json').write_text(json.dumps({**cfg, 'recipe_sha256':RECIPE_SHA,
        'labels':list(LABELS), 'target_source':'HYBRID_labels_scores.csv',
        'image_preprocessing':'none: load saved float16 tensors directly',
        'triplet_policy':'preserve all exported windows, including recorded series boundaries'}, indent=2))
    event('inputs_verified', shards=len(roots), training=len(train_ids), validation=len(val_ids),
          heldout_gold=len(gold_data), images='float16 [K,3,384,384], already normalized',
          gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], config=cfg)
    resume = None
    if cfg['resume_path']:
        resume = torch.load(cfg['resume_path'], map_location='cpu', weights_only=True)
        if resume['training_split'] != split or resume['training_config']['recipe_sha256'] != RECIPE_SHA:
            raise ValueError('Resume dataset/split differs')
        for key in ('epochs', 'k', 'seed', 'backbone_lr', 'fusion_lr', 'head_lr', 'label_sha256'):
            if resume['training_config'].get(key, cfg[key] if key=='label_sha256' else None) != cfg[key]:
                raise ValueError(f'Resume setting differs: {key}')
        model = load_fused_checkpoint(resume)
    else:
        import hashlib
        rad = asset('resnet-50-radimagenet-marwan', 'ResNet50.pt')
        with rad.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734':
            raise ValueError('RadImageNet asset hash differs from branch manifest')
        model = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                          rad, FusionConfig(gradient_checkpointing=False))
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg['backbone_lr'], cfg['fusion_lr'], cfg['head_lr']),
                                  weight_decay=cfg['weight_decay'])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype==torch.float16, init_scale=1024)
    gpu_count,batch_size,cfg['accumulate'] = batch_settings(torch.cuda.device_count(),cfg['studies_per_gpu'],cfg['effective_batch'])
    if resume:
        previous = resume['training_config']
        if resume.get('batch_size',2)*previous['accumulate'] != cfg['effective_batch']:
            raise ValueError('Resume effective optimizer batch differs')
    network = RaggedBagParallel(FullBagClassifier(model, cfg['chunk_size'], dtype), list(range(gpu_count)),cfg['studies_per_gpu'])
    val_loader = DataLoader(Subset(eval_data, val_idx), batch_size=1, num_workers=cfg['workers'],
                            pin_memory=True, prefetch_factor=1 if cfg['workers'] else None)
    updates = math.ceil(math.ceil(len(train_idx)/batch_size)/cfg['accumulate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs']*updates, eta_min=1e-7)
    epoch_start, batch_start, global_step, best = 0, 0, 0, float('inf')
    history = []
    carried_loss, carried_seen = 0., 0
    if resume:
        import shutil
        prior_best = Path(cfg['resume_path']).with_name('fusion_best.pt')
        if prior_best.is_file():
            shutil.copyfile(prior_best, 'fusion_best.pt')
        optimizer.load_state_dict(resume['optimizer'])
        if resume['scaler'] and scaler.is_enabled():
            scaler.load_state_dict(resume['scaler'])
        scheduler.load_state_dict(resume['scheduler'])
        epoch_start = resume['epoch_index']
        batch_start = resume_batch_position(resume,batch_size,len(train_idx))
        global_step, best, history = resume['global_step'], resume['best_validation'], resume['history']
        carried_loss, carried_seen = resume['epoch_running'], resume['epoch_seen']
        torch.set_rng_state(resume['torch_rng'])
        if len(resume['cuda_rng'])!=gpu_count:
            raise ValueError('Resume GPU count differs')
        torch.cuda.set_rng_state_all(resume['cuda_rng'])
        random.setstate(resume['python_rng'])
        del resume
    event('model_ready', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
          optimizer_groups=[dict(name=g['name'], lr=g['lr']) for g in optimizer.param_groups],
          batch_size=batch_size, k=cfg['k'], effective_batch=batch_size*cfg['accumulate'],
          amp=str(dtype), gradient_checkpointing='full-backbone chunks', chunk_size=cfg['chunk_size'],
          target_source='HYBRID_labels_scores.csv')

    def save(epoch, next_batch):
        ck = fused_checkpoint(model)
        ck.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), scheduler=scheduler.state_dict(),
                  epoch_index=epoch, next_batch=next_batch, global_step=global_step, best_validation=best,
                  epoch_running=running if next_batch else 0., epoch_seen=seen if next_batch else 0,
                  batch_size=batch_size,
                  next_study=min(next_batch*batch_size,len(train_idx)),
                  history=history, training_split=split, training_config={**cfg, 'recipe_sha256':RECIPE_SHA},
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate())
        atomic_save(ck, 'fusion_last.pt')
        event('checkpoint_saved', epoch_index=epoch, next_batch=next_batch, optimizer_steps=global_step)

    for epoch in range(epoch_start, cfg['epochs']):
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)
        order = torch.randperm(len(train_idx), generator=generator).tolist()
        loader = DataLoader(Subset(train_data, [train_idx[i] for i in order[batch_start*batch_size:]]), batch_size=batch_size,
                            num_workers=cfg['workers'], pin_memory=True,
                            collate_fn=collate_studies,
                            prefetch_factor=1 if cfg['workers'] else None)
        optimizer.zero_grad(set_to_none=True)
        running, seen = (carried_loss, carried_seen) if epoch==epoch_start else (0., 0)
        skipped_updates = 0
        batches = math.ceil(len(train_idx)/batch_size)
        event('epoch_started', epoch=epoch+1, batches=batches, resume_batch=batch_start)
        for batch_index, batch in enumerate(loader,start=batch_start):
            images = batch['images']
            y = batch['targets'].cuda(non_blocking=True)
            if any(not torch.isfinite(image).all() for image in images):
                raise ValueError('Non-finite saved training tensor')
            with torch.autocast('cuda', dtype=dtype):
                logits = network(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            weight = accumulation_weight(batch_index,batch_size,cfg['accumulate'],len(train_idx),len(y))
            scaler.scale(loss*weight).backward()
            running += float(loss.detach())*len(y); seen += len(y)
            del images, logits, loss
            if batch_index < 3:
                event('microbatch_complete', epoch=epoch+1, batch=batch_index+1, loss=running/seen,
                      peak_gpu_gib=[round(torch.cuda.max_memory_allocated(i)/2**30,2) for i in range(gpu_count)])
            boundary = (batch_index+1)%cfg['accumulate']==0 or batch_index+1==batches
            if boundary:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                if not torch.isfinite(grad_norm) and not scaler.is_enabled():
                    raise FloatingPointError('Non-finite gradients')
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                advanced = scaler.get_scale() >= before
                if advanced:
                    scheduler.step(); global_step += 1
                    skipped_updates = 0
                else:
                    skipped_updates += 1
                    event('amp_update_skipped', consecutive=skipped_updates, scale=scaler.get_scale())
                    if skipped_updates >= 8:
                        raise FloatingPointError('Eight consecutive non-finite AMP updates')
                optimizer.zero_grad(set_to_none=True)
                if global_step <= 3 or global_step%10==0:
                    event('training_progress', epoch=epoch+1, batch=batch_index+1, batches=batches,
                          optimizer_steps=global_step, loss=running/seen, gradient_norm=float(grad_norm),
                          hours=(time.monotonic()-started)/3600,
                          peak_gpu_gib=[round(torch.cuda.max_memory_allocated(i)/2**30,2) for i in range(gpu_count)])
                if advanced and (global_step==1 or global_step%cfg['save_every_steps']==0):
                    save(epoch, batch_index+1)
                if time.monotonic()-started > cfg['max_hours']*3600:
                    save(epoch, batch_index+1)
                    event('paused_time_budget', epoch=epoch+1, next_batch=batch_index+1, optimizer_steps=global_step,
                          resume_checkpoint='fusion_last.pt')
                    return
        batch_start = 0
        validation_loss = evaluate(model, val_loader, dtype, chunk=cfg['chunk_size'])
        history.append(dict(epoch=epoch+1, training_loss=running/seen, validation_loss=validation_loss,
                            optimizer_steps=global_step))
        if validation_loss < best:
            best = validation_loss
            ck = fused_checkpoint(model)
            ck.update(epoch=epoch+1, validation_loss=best, training_config=cfg, recipe_sha256=RECIPE_SHA)
            atomic_save(ck, 'fusion_best.pt')
        Path('history.json').write_text(json.dumps(history, indent=2))
        save(epoch+1, 0)
        event('epoch_complete', **history[-1], best_validation=best)
        if time.monotonic()-started > cfg['max_hours']*3600:
            event('paused_time_budget', epoch=epoch+1, optimizer_steps=global_step, resume_checkpoint='fusion_last.pt')
            return
    event('training_complete', epochs=cfg['epochs'], optimizer_steps=global_step, best_validation=best)


if __name__ == '__main__':
    main(globals().get('RUN_CONFIG'))
