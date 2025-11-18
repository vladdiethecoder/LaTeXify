"""Download helpers and dataset-specific ingest wiring."""

from __future__ import annotations

import hashlib
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import requests

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:  # pragma: no cover - imported lazily during tests
    snapshot_download = None
    hf_hub_download = None

ChunkSize = 1 << 20

DOCLAYNET_ASSETS = {
    "DocLayNet_core.zip": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
    "DocLayNet_extra.zip": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip",
}

DOCBANK_TEXT_ARCHIVE = "DocBank_500K_txt.zip"
DOCBANK_COCO_ARCHIVE = "MSCOCO_Format_Annotation.zip"
DOCBANK_REPO = "liminghao1630/DocBank"


def _hash_file(path: Path, *, algo: str = "sha256") -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(ChunkSize)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    sha256: str | None = None,
    max_retries: int = 5,
    chunk_size: int = ChunkSize,
) -> Path:
    """Download a URL to destination, supporting resumption + checksum."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        resume_from = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                mode = "ab" if resume_from else "wb"
                with temp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(destination)
            break
        except Exception:
            if attempt == max_retries:
                temp_path.unlink(missing_ok=True)
                raise
            time.sleep(min(60, 2**attempt))
    if not destination.exists():
        raise FileNotFoundError(f"Download failed for {url}")
    if sha256:
        digest = _hash_file(destination)
        if digest.lower() != sha256.lower():
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}. Expected {sha256}, got {digest}."
            )
    return destination


def extract_archive(path: Path, target_dir: Path) -> None:
    """Extract .zip or tar.* archives into target_dir."""

    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target_dir)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            archive.extractall(target_dir)
        return
    raise ValueError(f"Unsupported archive format: {path}")


def hf_snapshot(repo_id: str, destination: Path, *, revision: str = "main") -> Path:
    """Download a HuggingFace dataset snapshot into destination."""

    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required but not installed.")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    return destination


def gdrive_download(file_id_or_url: str, destination: Path) -> Path:
    """Download a file hosted on Google Drive."""

    if file_id_or_url.startswith("http"):
        url = file_id_or_url
    else:
        url = f"https://drive.google.com/uc?export=download&id={file_id_or_url}"
    return download_file(url, destination)


def _download_sources(manifest: Mapping[str, object], slug_root: Path) -> List[Path]:
    """Download every source entry defined in the manifest."""

    downloaded: List[Path] = []
    raw_sources = manifest.get("sources") or []
    sources: Sequence[MutableMapping[str, object]] = raw_sources  # type: ignore[assignment]
    slug_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        url = source["url"]
        filename = source.get("filename") or url.split("/")[-1]
        sha256 = source.get("sha256")
        extract = source.get("extract", False)
        target = slug_root / filename
        downloaded.append(download_file(url, target, sha256=sha256))
        if extract:
            destination = slug_root / (source.get("extract_dir") or filename.rsplit(".", 1)[0])
            extract_archive(target, destination)
    return downloaded


SPECIAL_DOWNLOADERS: Dict[str, Callable[[dict, Path], None]] = {}


def register_downloader(slug: str):
    def decorator(func: Callable[[dict, Path], None]) -> Callable[[dict, Path], None]:
        SPECIAL_DOWNLOADERS[slug] = func
        return func

    return decorator


@register_downloader("mathbridge")
def _download_mathbridge(manifest: dict, root: Path) -> None:
    repo = manifest.get("hf_dataset")
    if not repo:
        raise KeyError("mathbridge manifest missing 'hf_dataset'.")
    hf_snapshot(repo, root)


def register_http_archive_slugs(slugs: Iterable[str]) -> None:
    for slug in slugs:

        @register_downloader(slug)
        def _downloader(manifest: dict, root: Path, *, _slug=slug) -> None:
            del _slug  # unused; prevents late binding confusion
            _download_sources(manifest, root)


register_http_archive_slugs(
    [
        "arabic-printed-text",
        "funsd",
        "tablebank",
        "visual-tableqa",
        "mnist-database",
        "natural-environment-ocr",
        "street-view-house-numbers",
        "textocr",
    ]
)


@register_downloader("doclaynet")
def _download_doclaynet(manifest: dict, source_dir: Path) -> None:
    downloads = source_dir / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for filename, url in DOCLAYNET_ASSETS.items():
        archive = downloads / filename
        download_file(url, archive)
        extract_archive(archive, source_dir / filename.replace(".zip", ""))


@register_downloader("docbank")
def _download_docbank(manifest: dict, source_dir: Path) -> None:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to download DocBank.")
    cache = source_dir / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for filename in (DOCBANK_TEXT_ARCHIVE, DOCBANK_COCO_ARCHIVE):
        path = hf_hub_download(
            DOCBANK_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache),
            local_dir_use_symlinks=False,
        )
        extract_archive(Path(path), source_dir / filename.replace(".zip", ""))


@register_downloader("grotoap2")
def _download_grotoap2(manifest: dict, source_dir: Path) -> None:
    downloads = source_dir / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    target_dir = source_dir / "GROTOAP2"
    target_dir.mkdir(parents=True, exist_ok=True)
    urls = manifest.get("sources") or []
    if not urls:
        urls = [
            {"url": f"http://cermine.ceon.pl/grotoap2/GROTOAP2-{idx}.zip", "extract": True}
            for idx in range(1, 6)
        ]
    for entry in urls:
        archive_path = downloads / entry["url"].split("/")[-1]
        try:
            download_file(entry["url"], archive_path, sha256=entry.get("sha256"))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            print(f"[warn] {entry['url']}: HTTP {status}; skipping this shard.")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {entry['url']}: download failed ({exc}); skipping.")
            continue
        try:
            extract_archive(archive_path, target_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {archive_path.name}: extraction failed ({exc}); skipping.")


@register_downloader("publaynet")
def _download_publaynet(manifest: dict, source_dir: Path) -> None:
    repo = manifest.get("hf_dataset")
    if not repo:
        raise KeyError("publaynet manifest missing 'hf_dataset'.")
    hf_snapshot(repo, source_dir)


@register_downloader("rvl-cdip")
def _download_rvl_cdip(manifest: dict, source_dir: Path) -> None:
    repo = manifest.get("hf_dataset")
    if not repo:
        raise KeyError("rvl-cdip manifest missing 'hf_dataset'.")
    hf_snapshot(repo, source_dir)


@register_downloader("diachronic-document-dataset-ladas-2-0")
def _download_ladas(manifest: dict, source_dir: Path) -> None:
    repo = manifest.get("hf_dataset")
    if not repo:
        raise KeyError("LADaS manifest missing 'hf_dataset'.")
    hf_snapshot(repo, source_dir)


@register_downloader("textbook-question-answering-tqa")
def _download_tqa(manifest: dict, source_dir: Path) -> None:
    sources = manifest.get("sources")
    if not sources:
        raise KeyError("TQA manifest missing 'sources'.")
    _download_sources(manifest, source_dir)


@register_downloader("newspaper-navigator")
def _download_newspaper_navigator(manifest: dict, source_dir: Path) -> None:
    repo = manifest.get("hf_dataset")
    if not repo:
        raise KeyError("Newspaper Navigator manifest missing 'hf_dataset'.")
    hf_snapshot(repo, source_dir)


def ensure_downloaded(slug: str, manifest: dict, raw_root: Path) -> None:
    """Ensure a dataset has been fetched according to manifest instructions."""

    source_dir = raw_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    handler = SPECIAL_DOWNLOADERS.get(slug)
    if handler:
        handler(manifest, source_dir)
        return
    if manifest.get("hf_dataset"):
        hf_snapshot(manifest["hf_dataset"], source_dir)
        return
    if manifest.get("endpoint") or manifest.get("sources"):
        _download_sources(manifest, source_dir)
        return
    raise RuntimeError(f"No automated download handler registered for slug '{slug}'.")
