#!/usr/bin/env python3
"""Build deterministic FFHQ PCA assets for the Eigenfaces demo."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import pyarrow.ipc as arrow_ipc
from PIL import Image
from sklearn.decomposition import PCA

SEED = 20260902
SIZE = 128
SAMPLE_COUNT = 5000
MAX_COMPONENTS = 512
CHECKPOINTS = (128, 256, 512)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(record: dict, cache: Path) -> Path:
    target = cache / record["thumbnail"]["file_path"]
    expected = record["thumbnail"]["file_md5"]
    if target.exists() and md5_file(target) == expected:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = record["thumbnail"]["file_url"]
    with requests.Session() as session:
        for attempt in range(4):
            response = session.get(url, timeout=90)
            response.raise_for_status()
            payload = response.content
            if len(payload) < 8192:
                text = payload.decode("utf-8", errors="ignore")
                links = [html.unescape(link) for link in text.split('"') if "export=download" in link]
                if len(links) == 1:
                    url = requests.compat.urljoin(url, links[0])
                    continue
            target.write_bytes(payload)
            if md5_file(target) == expected:
                return target
            target.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Checksum mismatch for {target.name}")
    raise RuntimeError(f"Could not download {target.name}")


def download_hf_subset(selected_ids: list[str], cache: Path) -> dict[str, Path]:
    """Download only selected rows from an FFHQ 128 mirror when Drive is unavailable."""
    selected_ints = {int(key) for key in selected_ids}
    block_offsets = sorted({(index // 100) * 100 for index in selected_ints})
    row_urls: dict[int, str] = {}

    def fetch_block(offset: int) -> list[tuple[int, str]]:
        for attempt in range(8):
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "luethan2025/FFHQ-128x128",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": 100,
                },
                timeout=90,
            )
            if response.status_code == 429:
                time.sleep(min(30, 1.5 * (attempt + 1)))
                continue
            response.raise_for_status()
            return [
                (int(item["row_idx"]), item["row"]["image"]["src"])
                for item in response.json()["rows"]
                if int(item["row_idx"]) in selected_ints
            ]
        raise RuntimeError(f"Mirror API remained rate limited for block {offset}")

    print(f"Resolving {len(block_offsets)} mirror row blocks…", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_block, offset) for offset in block_offsets]
        for index, future in enumerate(as_completed(futures), start=1):
            row_urls.update(future.result())
            if index % 50 == 0:
                print(f"  blocks {index}/{len(block_offsets)}", flush=True)

    if len(row_urls) != len(selected_ids):
        raise RuntimeError(f"Resolved {len(row_urls)} of {len(selected_ids)} selected rows")

    mirror_dir = cache / "hf-thumbnails"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    def fetch_image(index: int) -> tuple[int, Path]:
        target = mirror_dir / f"{index:05d}.jpg"
        if target.exists():
            try:
                with Image.open(target) as image:
                    if image.size == (SIZE, SIZE):
                        return index, target
            except Exception:
                target.unlink(missing_ok=True)
        for attempt in range(4):
            response = requests.get(row_urls[index], timeout=90)
            if response.ok:
                target.write_bytes(response.content)
                try:
                    with Image.open(target) as image:
                        if image.size == (SIZE, SIZE):
                            return index, target
                except Exception:
                    pass
                target.unlink(missing_ok=True)
            if attempt == 3:
                response.raise_for_status()
        raise RuntimeError(f"Could not download mirror row {index}")

    downloaded: dict[str, Path] = {}
    print(f"Downloading/validating {len(selected_ids)} mirror thumbnails…", flush=True)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(fetch_image, int(key)) for key in selected_ids]
        for count, future in enumerate(as_completed(futures), start=1):
            index, path = future.result()
            downloaded[str(index)] = path
            if count % 250 == 0:
                print(f"  {count}/{len(selected_ids)}", flush=True)
    return downloaded


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return gray.astype(np.float32)


def load_arrow_matrix(arrow_dir: Path, selected_ids: list[str]) -> np.ndarray:
    selected = {int(key): position for position, key in enumerate(selected_ids)}
    matrix = np.empty((len(selected_ids), SIZE * SIZE), dtype=np.float32)
    found = 0
    global_index = 0
    files = sorted(arrow_dir.glob("*.arrow"))
    if not files:
        raise RuntimeError(f"No Arrow files found in {arrow_dir}")
    print(f"Reading selected thumbnails from {len(files)} Arrow files…", flush=True)
    for path in files:
        with path.open("rb") as handle:
            reader = arrow_ipc.open_stream(handle)
            for batch in reader:
                image_column = batch.column(0)
                for row in range(batch.num_rows):
                    position = selected.get(global_index)
                    if position is not None:
                        payload = image_column[row].as_py()["bytes"]
                        with Image.open(io.BytesIO(payload)) as image:
                            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                        gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
                        matrix[position] = gray.reshape(-1)
                        found += 1
                    global_index += 1
    if found != len(selected_ids):
        raise RuntimeError(f"Read {found} of {len(selected_ids)} selected rows")
    return matrix


def save_gray(path: Path, values: np.ndarray) -> None:
    pixels = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    image.save(path, optimize=True)


def align_portrait(source_path: Path, cache: Path) -> np.ndarray:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    # Manual landmark correction recorded for this fixed supplied portrait:
    # eye line is approximately +4.5 degrees and face center is near (457, 500).
    image = image.rotate(-4.5, resample=Image.Resampling.BICUBIC, center=(457, 500))
    image = image.crop((80, 80, 834, 834)).resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )
    aligned_path = cache / "aligned-input.png"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(aligned_path, optimize=True)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def write_float32(path: Path, values: np.ndarray) -> str:
    contiguous = np.asarray(values, dtype="<f4").reshape(-1)
    path.write_bytes(contiguous.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/eigenfaces"))
    parser.add_argument("--output", type=Path, default=Path("public/eigenfaces"))
    parser.add_argument("--hf-mirror", action="store_true")
    parser.add_argument("--arrow-dir", type=Path)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading FFHQ metadata…", flush=True)
    records = json.loads(args.metadata.read_text())
    training_ids = sorted(
        (key for key, value in records.items() if value["category"] == "training"),
        key=int,
    )
    selected_ids = sorted(random.Random(SEED).sample(training_ids, SAMPLE_COUNT), key=int)
    selected = [records[key] for key in selected_ids]

    if args.arrow_dir:
        matrix = load_arrow_matrix(args.arrow_dir, selected_ids)
    elif args.hf_mirror:
        downloaded = download_hf_subset(selected_ids, args.cache)
    else:
        print(f"Downloading/validating {SAMPLE_COUNT} thumbnails…", flush=True)
        downloaded: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = {
                executor.submit(download_one, record, args.cache): key
                for key, record in zip(selected_ids, selected)
            }
            for index, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                downloaded[key] = future.result()
                if index % 250 == 0:
                    print(f"  {index}/{SAMPLE_COUNT}", flush=True)

    if not args.arrow_dir:
        print("Decoding grayscale training matrix…", flush=True)
        matrix = np.empty((SAMPLE_COUNT, SIZE * SIZE), dtype=np.float32)
        for index, key in enumerate(selected_ids):
            matrix[index] = load_gray(downloaded[key]).reshape(-1)

    print("Fitting 512-component randomized PCA…", flush=True)
    pca = PCA(
        n_components=MAX_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
        iterated_power=4,
    )
    pca.fit(matrix)
    components = pca.components_.astype(np.float32)

    for index in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k_full = MAX_COMPONENTS
    for checkpoint in CHECKPOINTS:
        if cumulative[checkpoint - 1] >= 0.95:
            k_full = checkpoint
            break

    portrait = align_portrait(args.portrait, args.cache).reshape(-1)
    centered = portrait - pca.mean_.astype(np.float32)
    weights = components[:k_full] @ centered
    baseline = pca.mean_.astype(np.float32) + components[:k_full].T @ weights

    save_gray(args.output / "mean.png", pca.mean_.reshape(SIZE, SIZE))
    save_gray(args.output / "reconstruction.png", baseline.reshape(SIZE, SIZE))

    baseline_sha = write_float32(args.output / "baseline.f32", baseline)
    component_records = []
    for index in range(8):
        vector = components[index]
        amplitude = float(np.max(np.abs(vector))) or 1.0
        thumbnail = 0.5 + 0.5 * vector.reshape(SIZE, SIZE) / amplitude
        thumbnail_path = args.output / f"pc-{index + 1:02d}.png"
        vector_path = args.output / f"pc-{index + 1:02d}.f32"
        save_gray(thumbnail_path, thumbnail)
        component_records.append(
            {
                "index": index + 1,
                "eigenvalue": float(pca.explained_variance_[index]),
                "explainedVariance": float(pca.explained_variance_ratio_[index]),
                "baselineWeight": float(weights[index]),
                "baselineZ": float(weights[index] / math.sqrt(pca.explained_variance_[index])),
                "vector": f"/eigenfaces/{vector_path.name}",
                "vectorSha256": write_float32(vector_path, vector),
                "thumbnail": f"/eigenfaces/{thumbnail_path.name}",
            }
        )

    provenance = []
    for key, record in zip(selected_ids, selected):
        metadata = record["metadata"]
        provenance.append(
            {
                "id": int(key),
                "thumbnailMd5": record["thumbnail"]["file_md5"],
                "author": metadata.get("author", ""),
                "sourceUrl": metadata.get("photo_url", ""),
                "license": metadata.get("license", ""),
                "licenseUrl": metadata.get("license_url", ""),
            }
        )
    (args.cache / "provenance.json").write_text(json.dumps(provenance, indent=2))

    manifest = {
        "schemaVersion": 1,
        "dataset": "Flickr-Faces-HQ (FFHQ)",
        "datasetUrl": "https://github.com/NVlabs/ffhq-dataset",
        "seed": SEED,
        "sampleCount": SAMPLE_COUNT,
        "width": SIZE,
        "height": SIZE,
        "flatteningOrder": "row-major",
        "grayscale": "sRGB luminance 0.2126R + 0.7152G + 0.0722B",
        "kFull": k_full,
        "explainedVariance": {
            str(checkpoint): float(cumulative[checkpoint - 1])
            for checkpoint in CHECKPOINTS
        },
        "baseline": "/eigenfaces/baseline.f32",
        "baselineSha256": baseline_sha,
        "mean": "/eigenfaces/mean.png",
        "reconstruction": "/eigenfaces/reconstruction.png",
        "components": component_records,
        "portraitAlignment": {
            "rotationDegrees": -4.5,
            "rotationCenter": [457, 500],
            "crop": [80, 80, 834, 834],
            "sourceSize": [914, 1003],
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote assets with K_full={k_full}; variance@512={cumulative[511]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
