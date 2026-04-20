from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from PIL import Image

from dataset_studio.services.dataset_service import DatasetService, IMAGE_SUFFIXES


WORKSPACE_ROOT = Path("/workspace").resolve()


class EmbeddingExportService:
    def __init__(self, dataset_service: DatasetService) -> None:
        self.dataset_service = dataset_service

    def export_dataset_npz(
        self,
        *,
        raw_dataset_path: str,
        raw_model_path: str,
        raw_output_filename: str = "",
        per_class_limit: int = 20,
        batch_size: int = 32,
        selected_paths_by_class: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        dataset_root = self.dataset_service.ensure_dataset_root(raw_dataset_path)
        train_dir = dataset_root / "train"
        if not train_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset is missing a train split: {train_dir}")

        requested_model_path = self._resolve_workspace_path(raw_model_path)
        model_path = self._resolve_embedding_model_path(requested_model_path)
        metadata = self._load_embedding_metadata(model_path)
        preprocess = self._preprocess_config(metadata)

        items_by_class = self._collect_train_images_by_class(train_dir)
        if per_class_limit <= 0:
            raise HTTPException(status_code=400, detail="Per-class export limit must be at least 1.")
        effective_selection = self._resolve_effective_selection_paths_by_class(
            dataset_root=dataset_root,
            train_dir=train_dir,
            items_by_class=items_by_class,
            per_class_limit=per_class_limit,
            selected_paths_by_class=selected_paths_by_class or {},
        )
        items = self._flatten_export_items(
            items_by_class=items_by_class,
            effective_selection=effective_selection,
            require_non_empty=True,
        )
        if not items:
            raise HTTPException(status_code=400, detail=f"No training images found under: {train_dir}")

        output_dir = dataset_root / "db"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = self._resolve_output_filename(
            requested_model_path=requested_model_path,
            raw_output_filename=raw_output_filename,
            per_class_limit=per_class_limit,
        )
        output_path = output_dir / output_filename
        if output_path.exists():
            raise HTTPException(status_code=400, detail=f"Output NPZ already exists: {output_path}")
        selection_manifest = self._write_selection_manifest(
            dataset_root=dataset_root,
            requested_model_path=requested_model_path,
            resolved_model_path=model_path,
            output_filename=output_filename,
            per_class_limit=per_class_limit,
            effective_selection=effective_selection,
        )

        session = self._create_session(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        labels: list[str] = []
        paths: list[str] = []
        embeddings_batches: list[np.ndarray] = []
        embedding_dim: int | None = None

        for batch in self._batched(items, batch_size):
            batch_array = np.stack(
                [self._preprocess_image(path=item["path"], preprocess=preprocess) for item in batch],
                axis=0,
            )
            output = session.run([output_name], {input_name: batch_array})[0]
            embeddings = np.asarray(output, dtype=np.float32)
            if embeddings.ndim != 2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Expected 2D embeddings from ONNX model, got shape {embeddings.shape}.",
                )
            if embedding_dim is None:
                embedding_dim = int(embeddings.shape[1])
                self._validate_embedding_shape(embedding_dim=embedding_dim, metadata=metadata, model_path=model_path)
            labels.extend(item["label"] for item in batch)
            paths.extend(str(item["path"]) for item in batch)
            embeddings_batches.append(embeddings)

        embeddings_array = np.concatenate(embeddings_batches, axis=0) if embeddings_batches else np.zeros((0, 0), dtype=np.float32)
        collection_name = f"{dataset_root.name}__{requested_model_path.stem}"

        np.savez_compressed(
            output_path,
            collection_name=np.asarray([collection_name]),
            labels=np.asarray(labels),
            paths=np.asarray(paths),
            embeddings=embeddings_array,
            dataset_path=np.asarray([str(dataset_root)]),
            train_split_path=np.asarray([str(train_dir)]),
            model_path=np.asarray([str(requested_model_path)]),
            resolved_model_path=np.asarray([str(model_path)]),
            exported_at=np.asarray([datetime.now(timezone.utc).isoformat()]),
        )

        class_names = {item["label"] for item in items}
        customized_class_count = sum(1 for value in (selected_paths_by_class or {}).values() if value is not None)
        return {
            "dataset_path": str(dataset_root),
            "train_split_path": str(train_dir),
            "model_path": str(requested_model_path),
            "resolved_model_path": str(model_path),
            "output_filename": output_filename,
            "output_dir": str(output_dir),
            "output_path": str(output_path),
            "selection_manifest_path": selection_manifest["output_path"],
            "class_count": len(class_names),
            "image_count": len(items),
            "embedding_dim": int(embedding_dim or 0),
            "per_class_limit": int(per_class_limit),
            "customized_class_count": int(customized_class_count),
            "preprocess": preprocess,
        }

    def save_selection_manifest(
        self,
        *,
        raw_dataset_path: str,
        raw_model_path: str,
        raw_output_filename: str = "",
        per_class_limit: int = 20,
        selected_paths_by_class: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        dataset_root = self.dataset_service.ensure_dataset_root(raw_dataset_path)
        train_dir = dataset_root / "train"
        if not train_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset is missing a train split: {train_dir}")
        if per_class_limit <= 0:
            raise HTTPException(status_code=400, detail="Per-class export limit must be at least 1.")

        requested_model_path = self._resolve_workspace_path(raw_model_path)
        output_filename = self._resolve_output_filename(
            requested_model_path=requested_model_path,
            raw_output_filename=raw_output_filename,
            per_class_limit=per_class_limit,
        )
        items_by_class = self._collect_train_images_by_class(train_dir)
        effective_selection = self._resolve_effective_selection_paths_by_class(
            dataset_root=dataset_root,
            train_dir=train_dir,
            items_by_class=items_by_class,
            per_class_limit=per_class_limit,
            selected_paths_by_class=selected_paths_by_class or {},
        )
        manifest = self._write_selection_manifest(
            dataset_root=dataset_root,
            requested_model_path=requested_model_path,
            resolved_model_path=None,
            output_filename=output_filename,
            per_class_limit=per_class_limit,
            effective_selection=effective_selection,
        )

        return {
            "dataset_path": str(dataset_root),
            "model_path": str(requested_model_path),
            "output_filename": output_filename,
            "output_dir": str(dataset_root / "db"),
            "output_path": manifest["output_path"],
            "class_count": manifest["class_count"],
            "image_count": manifest["image_count"],
            "missing_class_count": manifest["missing_class_count"],
            "per_class_limit": int(per_class_limit),
            "saved_at": manifest["saved_at"],
        }

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if not path.is_relative_to(WORKSPACE_ROOT):
            raise HTTPException(status_code=400, detail="Only /workspace paths are supported for model export.")
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model path does not exist: {path}")
        return path

    def _resolve_embedding_model_path(self, requested_path: Path) -> Path:
        if requested_path.suffix.lower() == ".onnx":
            return requested_path

        if requested_path.suffix.lower() == ".pth":
            candidate = requested_path.with_suffix(".onnx")
            if candidate.is_file():
                return candidate
            raise HTTPException(
                status_code=400,
                detail=(
                    "For now NPZ export expects an ONNX embedding model, or a .pth path with an adjacent .onnx file. "
                    f"Could not find: {candidate}"
                ),
            )

        raise HTTPException(status_code=400, detail="Model path must point to a .onnx or .pth file.")

    def _resolve_output_filename(
        self,
        *,
        requested_model_path: Path,
        raw_output_filename: str,
        per_class_limit: int,
    ) -> str:
        default_name = f"{requested_model_path.stem}_gallery_{per_class_limit}.npz"
        candidate = (raw_output_filename or "").strip()
        if not candidate:
            return default_name

        if "/" in candidate or "\\" in candidate:
            raise HTTPException(status_code=400, detail="Export filename must be a plain filename, not a path.")
        candidate = Path(candidate).name
        if candidate in {"", ".", ".."}:
            raise HTTPException(status_code=400, detail="Export filename cannot be empty.")

        if not candidate.lower().endswith(".npz"):
            candidate = f"{candidate}.npz"

        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
        sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
        if not sanitized:
            raise HTTPException(status_code=400, detail="Export filename must include letters or numbers.")
        if not sanitized.lower().endswith(".npz"):
            sanitized = f"{sanitized}.npz"
        return sanitized

    def _load_embedding_metadata(self, model_path: Path) -> dict[str, Any]:
        candidates = sorted(model_path.parent.glob("*_embedding_meta.json"))
        if not candidates:
            return {}
        try:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid embedding metadata JSON: {candidates[0]}") from exc

    def _preprocess_config(self, metadata: dict[str, Any]) -> dict[str, Any]:
        preprocess_meta = metadata.get("inference_preprocess") if isinstance(metadata, dict) else None
        img_size = int(metadata.get("img_size", 224)) if isinstance(metadata, dict) else 224

        if not isinstance(preprocess_meta, dict):
            return {
                "resize": [img_size, img_size],
                "resize_short_side": None,
                "center_crop": None,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            }

        resize = preprocess_meta.get("resize")
        center_crop = preprocess_meta.get("center_crop")
        resize_short_side = preprocess_meta.get("resize_short_side")

        if isinstance(resize, list) and len(resize) == 2:
            resize = [int(resize[0]), int(resize[1])]
        else:
            resize = None

        return {
            "resize": resize,
            "resize_short_side": int(resize_short_side) if resize_short_side else None,
            "center_crop": int(center_crop) if center_crop else None,
            "normalize_mean": [float(x) for x in preprocess_meta.get("normalize_mean", [0.485, 0.456, 0.406])],
            "normalize_std": [float(x) for x in preprocess_meta.get("normalize_std", [0.229, 0.224, 0.225])],
        }

    def _collect_train_images_by_class(self, train_dir: Path) -> dict[str, list[Path]]:
        items: dict[str, list[Path]] = {}
        for class_dir in sorted(child for child in train_dir.iterdir() if child.is_dir()):
            items[class_dir.name] = list(self._iter_image_files(class_dir))
        return items

    def _resolve_effective_selection_paths_by_class(
        self,
        *,
        dataset_root: Path,
        train_dir: Path,
        items_by_class: Dict[str, list[Path]],
        per_class_limit: int,
        selected_paths_by_class: dict[str, list[str]],
    ) -> dict[str, list[Path]]:
        selection_by_class: dict[str, list[Path]] = {}

        unknown_classes = sorted(set(selected_paths_by_class.keys()) - set(items_by_class.keys()))
        if unknown_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Selected export classes are not present in the train split: {', '.join(unknown_classes[:8])}",
            )

        for class_name, class_paths in items_by_class.items():
            custom_paths = selected_paths_by_class.get(class_name)
            if custom_paths is None:
                chosen_paths = class_paths[:per_class_limit]
            else:
                chosen_paths = self._resolve_selected_paths_for_class(
                    dataset_root=dataset_root,
                    train_dir=train_dir,
                    class_name=class_name,
                    class_paths=class_paths,
                    requested_paths=custom_paths,
                    per_class_limit=per_class_limit,
                )
            selection_by_class[class_name] = chosen_paths

        return selection_by_class

    def _flatten_export_items(
        self,
        *,
        items_by_class: Dict[str, list[Path]],
        effective_selection: dict[str, list[Path]],
        require_non_empty: bool,
    ) -> list[dict[str, Any]]:
        export_items: list[dict[str, Any]] = []
        missing_classes: list[str] = []

        for class_name, class_paths in items_by_class.items():
            chosen_paths = effective_selection.get(class_name, [])
            if class_paths and not chosen_paths:
                missing_classes.append(class_name)
                continue
            for path in chosen_paths:
                export_items.append({"label": class_name, "path": path})

        if require_non_empty and missing_classes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Every product needs at least one selected train image before export. Missing classes: "
                    + ", ".join(missing_classes[:10])
                ),
            )

        return export_items

    def _write_selection_manifest(
        self,
        *,
        dataset_root: Path,
        requested_model_path: Path,
        resolved_model_path: Path | None,
        output_filename: str,
        per_class_limit: int,
        effective_selection: dict[str, list[Path]],
    ) -> dict[str, Any]:
        output_dir = dataset_root / "db"
        output_dir.mkdir(parents=True, exist_ok=True)
        selection_stem = Path(output_filename).stem
        output_path = output_dir / f"{selection_stem}_selection.json"
        saved_at = datetime.now(timezone.utc).isoformat()
        image_count = sum(len(paths) for paths in effective_selection.values())
        missing_class_count = sum(1 for paths in effective_selection.values() if not paths)

        payload = {
            "dataset_path": str(dataset_root),
            "model_path": str(requested_model_path),
            "resolved_model_path": str(resolved_model_path) if resolved_model_path else None,
            "output_filename": output_filename,
            "per_class_limit": int(per_class_limit),
            "saved_at": saved_at,
            "class_count": len(effective_selection),
            "image_count": image_count,
            "missing_class_count": missing_class_count,
            "selected_paths_by_class": {
                class_name: [str(path) for path in paths]
                for class_name, paths in effective_selection.items()
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "output_path": str(output_path),
            "saved_at": saved_at,
            "class_count": len(effective_selection),
            "image_count": image_count,
            "missing_class_count": missing_class_count,
        }

    def _resolve_selected_paths_for_class(
        self,
        *,
        dataset_root: Path,
        train_dir: Path,
        class_name: str,
        class_paths: list[Path],
        requested_paths: list[str],
        per_class_limit: int,
    ) -> list[Path]:
        if len(requested_paths) > per_class_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Class {class_name} has {len(requested_paths)} selected images, which exceeds the limit of {per_class_limit}.",
            )

        available = {str(path): path for path in class_paths}
        chosen: list[Path] = []
        seen: set[str] = set()

        for raw_path in requested_paths:
            path = self.dataset_service.resolve_data_path(raw_path)
            if not path.is_relative_to(train_dir / class_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Selected image does not belong to train/{class_name}: {path}",
                )
            path_key = str(path)
            if path_key not in available:
                raise HTTPException(
                    status_code=400,
                    detail=f"Selected image is not available for export in train/{class_name}: {path}",
                )
            if path_key in seen:
                continue
            seen.add(path_key)
            chosen.append(available[path_key])

        return chosen

    def _iter_image_files(self, directory: Path) -> Iterable[Path]:
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                yield child

    def _create_session(self, model_path: Path) -> ort.InferenceSession:
        available = ort.get_available_providers()
        providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        try:
            return ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load ONNX model: {model_path}") from exc

    def _preprocess_image(self, *, path: Path, preprocess: dict[str, Any]) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        resize = preprocess.get("resize")
        resize_short_side = preprocess.get("resize_short_side")
        center_crop = preprocess.get("center_crop")

        if resize:
            image = image.resize((int(resize[1]), int(resize[0])), resample=Image.Resampling.BILINEAR)
        elif resize_short_side:
            image = self._resize_short_side(image, int(resize_short_side))

        if center_crop:
            image = self._center_crop(image, int(center_crop))

        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(preprocess["normalize_mean"], dtype=np.float32)
        std = np.asarray(preprocess["normalize_std"], dtype=np.float32)
        arr = (arr - mean) / std
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    def _resize_short_side(self, image: Image.Image, short_side: int) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Encountered an invalid image with zero-sized dimensions.")

        if width <= height:
            new_width = short_side
            new_height = int(round(height * (short_side / width)))
        else:
            new_height = short_side
            new_width = int(round(width * (short_side / height)))
        return image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

    def _center_crop(self, image: Image.Image, crop_size: int) -> Image.Image:
        width, height = image.size
        left = max((width - crop_size) // 2, 0)
        top = max((height - crop_size) // 2, 0)
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom))

    def _validate_embedding_shape(self, *, embedding_dim: int, metadata: dict[str, Any], model_path: Path) -> None:
        num_classes = metadata.get("num_classes") if isinstance(metadata, dict) else None
        if isinstance(num_classes, int) and embedding_dim == num_classes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The resolved ONNX output dimension matches the class count, so it looks like a classifier/logit model "
                    f"instead of an embedding model: {model_path}"
                ),
            )

    def _batched(self, items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
        if batch_size <= 0:
            batch_size = 32
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]
