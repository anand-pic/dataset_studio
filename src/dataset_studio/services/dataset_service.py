from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi import HTTPException


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_SPLITS = ("train", "val", "test")
CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DATA_ROOT = Path("/workspace/vdata").resolve()
SKIP_DISCOVERY_DIRS = {
    ".dataset_studio",
    "__pycache__",
    "artifacts",
    "catalog",
    "cvat",
    "import_zips",
    "labels",
    "manifests",
    "product_labels",
    "product_preds",
    "raw",
    "runs",
    "shelf_labels",
    "shelf_preds",
    "tmp",
}


@dataclass(frozen=True)
class ClassImage:
    split: str
    class_name: str
    path: Path


class DatasetService:
    def resolve_data_path(self, raw_path: str, *, must_exist: bool = True) -> Path:
        path = Path(raw_path).expanduser()
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(DATA_ROOT):
            raise HTTPException(status_code=400, detail="Only /workspace/vdata paths are supported.")
        if must_exist and not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Path does not exist: {resolved}")
        return resolved

    def ensure_dataset_root(self, raw_path: str, *, allow_missing: bool = False) -> Path:
        root = self.resolve_data_path(raw_path, must_exist=not allow_missing)
        if root.exists() and not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Dataset path is not a directory: {root}")
        if root.exists() and not self.looks_like_dataset_root(root):
            raise HTTPException(status_code=400, detail=f"Path is not a recognition dataset root: {root}")
        return root

    def discover_datasets(self, max_depth: int = 6) -> List[dict]:
        discovered: List[dict] = []
        stack = [(DATA_ROOT, 0)]
        seen: set[Path] = set()

        while stack:
            current, depth = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if depth > max_depth or not current.is_dir():
                continue
            if self.looks_like_dataset_root(current):
                discovered.append(self._scan_summary(current))
                continue

            try:
                children = sorted((child for child in current.iterdir() if child.is_dir()), key=lambda p: p.name)
            except PermissionError:
                continue

            for child in reversed(children):
                if child.name.startswith("."):
                    continue
                if depth >= 2 and child.name in SKIP_DISCOVERY_DIRS:
                    continue
                stack.append((child, depth + 1))

        discovered.sort(key=lambda item: item["path"])
        return discovered

    def scan_dataset(self, raw_path: str) -> dict:
        root = self.ensure_dataset_root(raw_path)
        return self._scan_full(root)

    def list_class_images(self, raw_dataset_path: str, class_name: str) -> Dict[str, List[Path]]:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        self._validate_class_name(class_name)

        files_by_split: Dict[str, List[Path]] = {}
        for split in self.available_splits(dataset_root):
            class_dir = dataset_root / split / class_name
            if not class_dir.is_dir():
                continue
            files_by_split[split] = list(self._iter_image_files(class_dir))
        return files_by_split

    def get_class_detail(self, raw_dataset_path: str, class_name: str, limit_per_split: int = 120) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        self._validate_class_name(class_name)

        splits = []
        total = 0
        for split in self.available_splits(dataset_root):
            class_dir = dataset_root / split / class_name
            if not class_dir.is_dir():
                continue

            files = list(self._iter_image_files(class_dir))
            total += len(files)
            entries = []
            for file_path in files[:limit_per_split]:
                entries.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "split": split,
                        "class_name": class_name,
                        "size_bytes": file_path.stat().st_size,
                        "source_hint": self._source_hint_from_filename(file_path.name),
                    }
                )

            splits.append(
                {
                    "split": split,
                    "count": len(files),
                    "truncated": len(files) > limit_per_split,
                    "images": entries,
                }
            )

        if total == 0:
            raise HTTPException(status_code=404, detail=f"Class not found: {class_name}")

        return {
            "dataset_path": str(dataset_root),
            "class_name": class_name,
            "total": total,
            "splits": splits,
        }

    def rename_class(self, raw_dataset_path: str, old_name: str, new_name: str) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        self._validate_class_name(old_name)
        self._validate_class_name(new_name)
        if old_name == new_name:
            return {"renamed": False, "detail": "Class name unchanged."}

        touched = 0
        moved = 0
        for split in self.available_splits(dataset_root):
            old_dir = dataset_root / split / old_name
            if not old_dir.is_dir():
                continue
            touched += 1

            new_dir = dataset_root / split / new_name
            if new_dir.exists():
                new_dir.mkdir(parents=True, exist_ok=True)
                for source_file in self._iter_image_files(old_dir):
                    target_file = self._build_unique_path(new_dir / source_file.name)
                    shutil.move(str(source_file), str(target_file))
                    moved += 1
                old_dir.rmdir()
            else:
                old_dir.rename(new_dir)
                moved += sum(1 for _ in self._iter_image_files(new_dir))

        if touched == 0:
            raise HTTPException(status_code=404, detail=f"Class not found: {old_name}")

        return {
            "renamed": True,
            "old_name": old_name,
            "new_name": new_name,
            "split_count": touched,
            "moved_images": moved,
        }

    def reassign_image(self, raw_dataset_path: str, raw_image_path: str, target_class: str, target_split: str) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        self._validate_class_name(target_class)
        target_split = self._validate_split(target_split)

        image_path, current_split, current_class = self._resolve_dataset_image(dataset_root, raw_image_path)
        target_path = self._reassign_image_path(
            dataset_root=dataset_root,
            image_path=image_path,
            target_class=target_class,
            target_split=target_split,
        )
        if target_path == image_path:
            return {"moved": False, "detail": "Image already in the requested location."}

        return {
            "moved": True,
            "from_split": current_split,
            "from_class": current_class,
            "to_split": target_split,
            "to_class": target_class,
            "target_path": str(target_path),
        }

    def trash_image(self, raw_dataset_path: str, raw_image_path: str) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        image_path, _, _ = self._resolve_dataset_image(dataset_root, raw_image_path)
        trash_path = self._trash_image_path(dataset_root=dataset_root, image_path=image_path)

        return {
            "trashed": True,
            "trash_path": str(trash_path),
        }

    def reassign_images(
        self,
        raw_dataset_path: str,
        raw_image_paths: List[str],
        target_class: str,
        target_split: str,
    ) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        self._validate_class_name(target_class)
        target_split = self._validate_split(target_split)
        image_paths = self._dedupe_requested_paths(raw_image_paths)

        moved = 0
        skipped = 0
        for raw_image_path in image_paths:
            image_path, _, _ = self._resolve_dataset_image(dataset_root, raw_image_path)
            target_path = self._reassign_image_path(
                dataset_root=dataset_root,
                image_path=image_path,
                target_class=target_class,
                target_split=target_split,
            )
            if target_path == image_path:
                skipped += 1
            else:
                moved += 1

        return {
            "moved": moved,
            "skipped": skipped,
            "target_class": target_class,
            "target_split": target_split,
        }

    def trash_images(self, raw_dataset_path: str, raw_image_paths: List[str]) -> dict:
        dataset_root = self.ensure_dataset_root(raw_dataset_path)
        image_paths = self._dedupe_requested_paths(raw_image_paths)

        trashed = 0
        trash_paths: List[str] = []
        for raw_image_path in image_paths:
            image_path, _, _ = self._resolve_dataset_image(dataset_root, raw_image_path)
            trash_path = self._trash_image_path(dataset_root=dataset_root, image_path=image_path)
            trash_paths.append(str(trash_path))
            trashed += 1

        return {
            "trashed": trashed,
            "trash_paths": trash_paths,
        }

    def available_splits(self, dataset_root: Path) -> List[str]:
        return [split for split in KNOWN_SPLITS if (dataset_root / split).is_dir()]

    def looks_like_dataset_root(self, path: Path) -> bool:
        if not path.is_dir():
            return False

        split_dirs = [path / split for split in KNOWN_SPLITS if (path / split).is_dir()]
        if not split_dirs:
            return False

        for split_dir in split_dirs:
            try:
                children = [child for child in split_dir.iterdir() if child.is_dir()]
            except PermissionError:
                continue
            class_children = [child for child in children if child.name not in {"images", "labels"}]
            if class_children:
                return True
        return False

    def _scan_summary(self, root: Path) -> dict:
        summary = self._scan_full(root, include_class_entries=False)
        return {
            "path": summary["path"],
            "relative_path": summary["relative_path"],
            "kind": summary["kind"],
            "class_count": summary["class_count"],
            "total_images": summary["total_images"],
            "splits": summary["splits"],
        }

    def _scan_full(self, root: Path, *, include_class_entries: bool = True) -> dict:
        class_map: Dict[str, dict] = {}
        split_totals: Dict[str, int] = {}
        total_images = 0

        for split in self.available_splits(root):
            split_total = 0
            split_dir = root / split
            for class_dir in self._iter_class_dirs(split_dir):
                class_name = class_dir.name
                file_count = 0
                sample_path = None
                for image_path in self._iter_image_files(class_dir):
                    if sample_path is None:
                        sample_path = image_path
                    file_count += 1
                if file_count == 0:
                    continue

                entry = class_map.setdefault(
                    class_name,
                    {
                        "name": class_name,
                        "counts": {name: 0 for name in KNOWN_SPLITS},
                        "total": 0,
                        "sample_path": None,
                    },
                )
                entry["counts"][split] = file_count
                entry["total"] += file_count
                if entry["sample_path"] is None and sample_path is not None:
                    entry["sample_path"] = str(sample_path)
                split_total += file_count

            split_totals[split] = split_total
            total_images += split_total

        classes = sorted(class_map.values(), key=lambda item: self._class_sort_key(item["name"]))
        if not include_class_entries:
            classes = []

        return {
            "path": str(root),
            "relative_path": str(root.relative_to(DATA_ROOT)),
            "kind": self._dataset_kind(root),
            "class_count": len(class_map),
            "total_images": total_images,
            "splits": split_totals,
            "classes": classes,
        }

    def _iter_class_dirs(self, split_dir: Path) -> Iterable[Path]:
        if not split_dir.is_dir():
            return []
        return sorted(
            (child for child in split_dir.iterdir() if child.is_dir() and child.name not in {"images", "labels"}),
            key=lambda p: self._class_sort_key(p.name),
        )

    def _iter_image_files(self, directory: Path) -> Iterable[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            (child for child in directory.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: p.name,
        )

    def _validate_class_name(self, name: str) -> str:
        name = name.strip()
        if not name or not CLASS_NAME_RE.fullmatch(name):
            raise HTTPException(
                status_code=400,
                detail="Class names must use only letters, numbers, dot, underscore, or dash.",
            )
        return name

    def _validate_split(self, split: str) -> str:
        split = split.strip().lower()
        if split not in KNOWN_SPLITS:
            raise HTTPException(status_code=400, detail=f"Invalid split: {split}")
        return split

    def _build_unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        index = 2
        while True:
            candidate = path.with_name(f"{stem}__dup{index}{suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    def _cleanup_if_empty(self, directory: Path) -> None:
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    def _resolve_dataset_image(self, dataset_root: Path, raw_image_path: str) -> tuple[Path, str, str]:
        image_path = self.resolve_data_path(raw_image_path)
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
        if not image_path.is_relative_to(dataset_root):
            raise HTTPException(status_code=400, detail="Image does not belong to the selected dataset.")

        parts = image_path.relative_to(dataset_root).parts
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Image path is not inside a split/class directory.")

        current_split, current_class = parts[0], parts[1]
        self._validate_split(current_split)
        return image_path, current_split, current_class

    def _reassign_image_path(self, *, dataset_root: Path, image_path: Path, target_class: str, target_split: str) -> Path:
        target_dir = dataset_root / target_split / target_class
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._build_unique_path(target_dir / image_path.name)
        if target_path == image_path:
            return image_path

        shutil.move(str(image_path), str(target_path))
        self._cleanup_if_empty(image_path.parent)
        return target_path

    def _trash_image_path(self, *, dataset_root: Path, image_path: Path) -> Path:
        relative = image_path.relative_to(dataset_root)
        trash_root = dataset_root / ".dataset_studio" / "trash" / self._timestamp_id()
        trash_dir = trash_root / relative.parent
        trash_dir.mkdir(parents=True, exist_ok=True)
        trash_path = self._build_unique_path(trash_dir / image_path.name)
        shutil.move(str(image_path), str(trash_path))
        self._cleanup_if_empty(image_path.parent)
        return trash_path

    def _dedupe_requested_paths(self, raw_image_paths: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for raw_path in raw_image_paths:
            value = raw_path.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        if not deduped:
            raise HTTPException(status_code=400, detail="Select at least one image first.")
        return deduped

    def _class_sort_key(self, value: str) -> tuple:
        if value.isdigit():
            return (0, int(value), value)
        return (1, value.lower(), value)

    def _dataset_kind(self, root: Path) -> str:
        parts = root.parts
        if "outputs" in parts and "recognition_dataset" in parts:
            return "recognition_export"
        return "working_dataset"

    def _source_hint_from_filename(self, filename: str) -> str | None:
        if "__" not in filename:
            return None
        return filename.split("__", 1)[0]

    def _timestamp_id(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
