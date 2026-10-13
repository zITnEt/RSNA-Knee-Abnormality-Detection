"""One-off patch: fast mid-epoch resume and post-training validation of extra checkpoints.

    python scripts/patch_resume_validation.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path, replacements, marker):
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    if marker in text:
        print(path.name, 'already patched', hashlib.sha256(text.encode()).hexdigest())
        return
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f'{path.name}: expected one match for {old[:70]!r}, found {text.count(old)}')
        text = text.replace(old, new)
    path.write_bytes(text.encode('utf-8'))
    print(path.name, hashlib.sha256(text.encode()).hexdigest())


patch(ROOT/'scripts/train_fusion_ddp.py', [
    # Resume mid-epoch without loading the already-trained studies from disk.
    ("""        mine = [train_idx[i] for i in rank_shard(order)]
        loader = DataLoader(Subset(train_data, mine), batch_size=1, num_workers=cfg['workers'], pin_memory=True,
                            prefetch_factor=2 if cfg['workers'] else None)""",
     """        mine = [train_idx[i] for i in rank_shard(order)]
        n_mb = len(mine)
        loader = DataLoader(Subset(train_data, mine[batch_start:]), batch_size=1, num_workers=cfg['workers'],
                            pin_memory=True, prefetch_factor=2 if cfg['workers'] else None)"""),
    ("""        event('epoch_started', epoch=epoch+1, microbatches_per_rank=len(loader), resume_batch=batch_start)""",
     """        event('epoch_started', epoch=epoch+1, microbatches_per_rank=n_mb, resume_batch=batch_start)"""),
    ("""        for batch_index, batch in enumerate(loader):
            if batch_index < batch_start:
                fetch_start = time.monotonic()
                continue
            got""",
     """        for batch_index, batch in enumerate(loader, start=batch_start):
            got"""),
    ("group_size = min(cfg['accumulate'], len(loader)-group_start)",
     "group_size = min(cfg['accumulate'], n_mb-group_start)"),
    ("boundary = (batch_index+1) % cfg['accumulate'] == 0 or batch_index+1 == len(loader)",
     "boundary = (batch_index+1) % cfg['accumulate'] == 0 or batch_index+1 == n_mb"),
    ("remaining = (len(loader)-batch_index-1) + (cfg['epochs']-epoch-1)*len(loader)",
     "remaining = (n_mb-batch_index-1) + (cfg['epochs']-epoch-1)*n_mb"),
    ("batch=batch_index+1, batches=len(loader),", "batch=batch_index+1, batches=n_mb,"),
    ("projected_epoch_minutes=round(per_mb*len(loader)/60, 1),", "projected_epoch_minutes=round(per_mb*n_mb/60, 1),"),
    ("""    finished = False
    for epoch in range(epoch_start, cfg['epochs']):""",
     """    finished = stop = False
    for epoch in range(epoch_start, cfg['epochs']):"""),
    # Post-training validation of extra weights (e.g. the gold-selected best checkpoint and the parent).
    ("""    if finished:
        event('training_complete', optimizer_steps=global_step, best=best,
              hours=round((time.monotonic()-started)/3600, 3))
    if WORLD > 1:""",
     """    if finished:
        event('training_complete', optimizer_steps=global_step, best=best,
              hours=round((time.monotonic()-started)/3600, 3))
    if not stop and (cfg.get('extra_checkpoints') or cfg.get('validate_parent')):
        final = {}
        if cfg.get('validate_parent'):
            import hashlib
            rad = asset('resnet-50-radimagenet-marwan', 'ResNet50.pt')
            with rad.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != RAD_SHA:
                    raise ValueError('RadImageNet asset hash differs from branch manifest')
            parent = build_raptor_v5_radimagenet(asset('raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'),
                                                 rad, FusionConfig(gradient_checkpointing=False))
            weights = [('parent_step0', parent.state_dict())]
            del parent
        else:
            weights = []
        for name, path in (cfg.get('extra_checkpoints') or {}).items():
            weights.append((name, torch.load(path, map_location='cpu', weights_only=True)['model']))
        for name, state in weights:
            model.load_state_dict(state, strict=True)
            final[name] = run_evaluation(f'final_{name}', model, {**gold_sets, **validation_sets}, truth, cfg,
                                         dtype, global_step, device)
            event('final_validation', model=name,
                  macro_auc=None if final[name] is None else {k: v['macro_auc'] for k, v in final[name].items()})
        if RANK == 0:
            Path('final_validation.json').write_text(json.dumps(final, indent=2))
    if WORLD > 1:"""),
], marker='n_mb = len(mine)')

patch(ROOT/'scripts/build_fusion_v2_notebook.py', [
    ("""        launch = ('import json, os, subprocess, sys, torch\\n'""",
     """        launch = ('import json, os, subprocess, sys, torch\\n'
                  'from pathlib import Path as _P\\n'
                  'def _find(name):\\n'
                  "    hits = sorted(p for base in ('/kaggle/input/notebooks', '/kaggle/input') for p in _P(base).glob(f'*/*/{name}') if 'rsna-fusion-ddp-train' in str(p))\\n"
                  "    hits = hits or sorted(p for p in _P('/kaggle/input').glob(f'*/{name}') if 'rsna-fusion-ddp-train' in str(p))\\n"
                  "    assert len(hits) == 1, (name, hits)\\n"
                  '    return str(hits[0])\\n'
                  "if RUN_CONFIG.get('resume_path') == 'auto':\\n"
                  "    RUN_CONFIG['resume_path'] = _find('fusion_last.pt')\\n"
                  "for key, value in list((RUN_CONFIG.get('extra_checkpoints') or {}).items()):\\n"
                  "    if value.startswith('auto:'):\\n"
                  "        RUN_CONFIG['extra_checkpoints'][key] = _find(value[5:])\\n"
                  "print('RUN_CONFIG', RUN_CONFIG, flush=True)\\n\""""),
    ("""                    dataset_sources=DATASETS, kernel_sources=PREPROCESSING, competition_sources=[],""",
     """                    dataset_sources=DATASETS, kernel_sources=PREPROCESSING + list(extra_kernels), competition_sources=[],"""),
    ("def build(slug, smoke, run_config, launcher='single'):",
     "def build(slug, smoke, run_config, launcher='single', extra_kernels=()):"),
    ("    parser.add_argument('--launcher', choices=('single', 'ddp'), default='single')\n",
     "    parser.add_argument('--launcher', choices=('single', 'ddp'), default='single')\n"
     "    parser.add_argument('--extra-kernel', action='append', default=[])\n"),
    ("    notebook, metadata = build(args.slug, args.smoke, json.loads(args.config), args.launcher)",
     "    notebook, metadata = build(args.slug, args.smoke, json.loads(args.config), args.launcher, args.extra_kernel)"),
], marker='extra_kernels')
