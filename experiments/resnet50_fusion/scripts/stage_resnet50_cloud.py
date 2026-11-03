"""Download the exact private Kaggle V2 shards on the training VM, never locally."""
import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(path, directory):
    directory = directory.resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (directory / info.filename).resolve()
            if not target.is_relative_to(directory):
                raise ValueError('Archive member outside staging directory')
        archive.extractall(directory)
    path.unlink()


def download_v2(api, shard, data, force=False):
    # Kaggle 1.7.x passes the '/2' string into an integer-only SDK field.
    # Set the version explicitly so Python 3.10 and newer clients agree.
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    slug = f'rsna-series-safe-384-shard-{shard:02d}'
    with api.build_kaggle_client() as client:
        request = ApiDownloadDatasetRequest()
        request.owner_slug = 'zafarnematov'
        request.dataset_slug = slug
        request.dataset_version_number = 2
        response = client.datasets.dataset_api_client.download_dataset(request)
        archive = str(data / (slug + '.zip'))
        if force or api.download_needed(response, archive, False):
            api.download_file(response, archive, client.http_client(), False, not force)


def download_v2_from_kernel(api, shard, data):
    """Recover the same V2 bytes from the notebook that published the dataset."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    prefix = f'min_overlap_384_shard_{shard:02d}/'
    root = data.resolve()
    token = None
    count = 0

    def fetch(item):
        if not item.file_name.startswith(prefix):
            raise ValueError(f'Unexpected publication output: {item.file_name}')
        target = (root / item.file_name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Publication output outside staging directory: {item.file_name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            return
        partial = target.with_name(target.name + '.partial')
        for attempt in range(5):
            try:
                with requests.get(item.url, stream=True, timeout=(30, 120)) as download:
                    download.raise_for_status()
                    with partial.open('wb') as stream:
                        for chunk in download.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                partial.replace(target)
                return
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(min(60, 10 * (attempt + 1)))

    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = 'zafarnematov'
            request.kernel_slug = f'rsna-series-safe-384-preprocessing-{shard:02d}'
            request.page_size = 200
            request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            items = response.files or []
            with ThreadPoolExecutor(max_workers=6) as pool:
                for future in [pool.submit(fetch, item) for item in items]:
                    future.result()
            count += len(items)
            token = response.next_page_token
            print(f'KERNEL_OUTPUT_V2_SHARD {shard:02d} files={count}', flush=True)
            if not token:
                break
    if not count:
        raise ValueError(f'No publication output for shard {shard:02d}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--expected', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Dataset staging must run on the Linux training VM')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    data, expected, assets = Path(args.data), Path(args.expected), Path(args.assets)
    data.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    manifests = sorted(expected.glob('shard_*.json'))
    if len(manifests) != 32:
        raise ValueError('Expected all 32 pinned publication manifests')
    needed = 0
    for source in manifests:
        manifest = json.loads(source.read_text())
        root = data / f"min_overlap_384_shard_{manifest['shard']:02d}"
        needed += sum(r['bytes'] for r in manifest['completed'] if not (root / r['file']).is_file())
    # Retained raw/EMA/periodic snapshots require approximately 80 GB beyond the dataset.
    free = shutil.disk_usage(data).free
    if free < needed + 90*2**30:
        raise RuntimeError(f'Need {(needed+90*2**30)/2**30:.1f} GiB free, have {free/2**30:.1f}; no existing data removed')
    for source in manifests:
        manifest = json.loads(source.read_text())
        shard = manifest['shard']
        root = data / f'min_overlap_384_shard_{shard:02d}'
        marker = root / '.download_verified.json'
        expected_sha = sha(source)
        verified = marker.exists() and json.loads(marker.read_text()).get('manifest_sha256') == expected_sha
        present = all((root/r['file']).is_file() and (root/r['file']).stat().st_size == r['bytes'] for r in manifest['completed'])
        if not present:
            print(f'DOWNLOAD_V2_SHARD {shard:02d}', flush=True)
            for attempt in range(5):
                try:
                    download_v2(api, shard, data, force=attempt > 0)
                    unpack(data / f'rsna-series-safe-384-shard-{shard:02d}.zip', data)
                    break
                except Exception as exc:
                    print(f'DOWNLOAD_RETRY {shard:02d} attempt={attempt+1} type={type(exc).__name__}', flush=True)
                    if getattr(getattr(exc, 'response', None), 'status_code', None) == 404:
                        print(f'DOWNLOAD_FROM_PUBLICATION_OUTPUT {shard:02d}', flush=True)
                        download_v2_from_kernel(api, shard, data)
                        break
                    if attempt == 4:
                        raise
                    time.sleep(min(60, 10*(attempt+1)))
            verified = False
        actual = json.loads((root/'manifest.json').read_text())
        if actual != manifest:
            raise ValueError(f'Shard {shard}: downloaded manifest differs from pinned V2')
        if not verified:
            for row in manifest['completed']:
                file = root/row['file']
                if not file.is_file() or file.stat().st_size != row['bytes'] or sha(file) != row['sha256']:
                    raise ValueError(f'Corrupt or wrong V2 study: {file.name}')
            marker.write_text(json.dumps(dict(manifest_sha256=expected_sha, verified=True)))
        print(f'SHARD_VERIFIED {shard:02d} {len(manifest["completed"])}', flush=True)
    raptor = assets/'raptor_ft_coatnet_v5_full_swa.pt'
    if not raptor.exists():
        api.dataset_download_file('dreaddevelopment/raptor-knee-maxspan', raptor.name,
                                  path=str(assets), quiet=False)
        if not raptor.exists() and (assets/(raptor.name+'.zip')).exists():
            unpack(assets/(raptor.name+'.zip'), assets)
    if not raptor.exists():
        raise FileNotFoundError('Raptor pretrained checkpoint download did not produce the expected file')
    resnet = assets/'resnet50-11ad3fa6.pth'
    if not resnet.exists():
        from torch.hub import download_url_to_file
        download_url_to_file('https://download.pytorch.org/models/resnet50-11ad3fa6.pth', str(resnet), hash_prefix='11ad3fa6')
    if not sha(resnet).startswith('11ad3fa6'):
        raise ValueError('ImageNet ResNet50 checkpoint hash mismatch')
    print('ALL_DATA_AND_PARENT_ASSETS_READY', flush=True)


if __name__ == '__main__':
    main()
