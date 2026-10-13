"""Build a private synthetic T4 benchmark; no medical input or parent assets."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    model=(ROOT/'models/raptor_v5_radimagenet.py').read_text(encoding='utf-8')
    trainer=(ROOT/'scripts/train_fusion_ready.py').read_text(encoding='utf-8').split("if __name__ == '__main__':")[0]
    trainer='\n'.join(line for line in trainer.splitlines() if not line.startswith(
        ('from models.raptor_v5_radimagenet import ','from raptor_data.ready_tensors import ')))
    benchmark=(ROOT/'scripts/benchmark_fusion_batches.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
    benchmark='\n'.join(line for line in benchmark.splitlines() if not line.startswith(
        ('ROOT=','sys.path.insert','from models.raptor_v5_radimagenet import ','from scripts.train_fusion_ready import ')))
    benchmark=benchmark.replace("directory=ROOT/'results/fusion_training_hybrid50/benchmarks'",
                                "directory=Path('/kaggle/working/benchmarks')")
    setup="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','--no-deps','timm==1.0.22'])\n"
    run='''
for c,s in [(4,1),(8,2),(12,2)]:
    try:
        main(c,s,2)
    except torch.OutOfMemoryError:
        print(json.dumps(dict(event='out_of_memory',chunk_size=c,studies_per_gpu=s)),flush=True)
    gc.collect()
    torch.cuda.empty_cache()
'''
    cells=[]
    for i,source in enumerate([setup,model,trainer,benchmark+run]):
        ast.parse(source)
        cells.append(dict(cell_type='code',id=f'bench-{i}',metadata={},source=source.splitlines(keepends=True),
                          outputs=[],execution_count=None))
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={
        'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
    folder=ROOT/'kaggle/fusion_batch_benchmark'; folder.mkdir(exist_ok=True)
    (folder/'benchmark.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata=dict(id='zafarnematov/rsna-fusion-batch-throughput-probe',title='RSNA Fusion Batch Throughput Probe',
        code_file='benchmark.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
