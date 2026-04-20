from __future__ import annotations

import filecmp
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, Iterable, List
from uuid import uuid4

from fastapi import HTTPException

from dataset_studio.services.dataset_service import DATA_ROOT, DatasetService, KNOWN_SPLITS


class MergeService:
    def __init__(self, dataset_service: DatasetService) -> None:
        self.dataset_service = dataset_service

    def build_preview(self, target_path: str, source_path: str) -> dict:
        source_root = self.dataset_service.ensure_dataset_root(source_path)
        target_root = self.dataset_service.ensure_dataset_root(target_path, allow_missing=True)

        source_scan = self.dataset_service.scan_dataset(str(source_root))
        target_scan = self._scan_target_for_preview(target_root)
        target_classes = {entry["name"] for entry in target_scan.get("classes", [])}
        source_tag = self._derive_source_tag(source_root)

        mappings = []
        for source_entry in source_scan["classes"]:
            source_class = source_entry["name"]
            if source_class in target_classes:
                target_class = source_class
                action = "merge"
            elif source_class.startswith("unknown_"):
                target_class = f"{source_tag}__{source_class}"
                action = "create_placeholder"
            else:
                target_class = source_class
                action = "create"

            mappings.append(
                {
                    "source_class": source_class,
                    "target_class": target_class,
                    "enabled": True,
                    "action": action,
                    "source_counts": source_entry["counts"],
                    "source_total": source_entry["total"],
                    "target_exists": target_class in target_classes,
                    "sample_path": source_entry.get("sample_path"),
                }
            )

        return {
            "source": source_scan,
            "target": target_scan,
            "source_tag": source_tag,
            "mappings": mappings,
        }

    def commit_merge(self, target_path: str, source_path: str, mappings: List[dict]) -> dict:
        source_root = self.dataset_service.ensure_dataset_root(source_path)
        target_root = self.dataset_service.ensure_dataset_root(target_path, allow_missing=True)
        source_scan = self.dataset_service.scan_dataset(str(source_root))

        if target_root.exists() and target_root.is_file():
            raise HTTPException(status_code=400, detail=f"Target path is a file: {target_root}")
        target_root.mkdir(parents=True, exist_ok=True)

        source_classes = {entry["name"] for entry in source_scan["classes"]}
        mapping_by_source: Dict[str, dict] = {}
        enabled_count = 0
        for mapping in mappings:
            source_class = mapping["source_class"].strip()
            if source_class not in source_classes:
                raise HTTPException(status_code=400, detail=f"Unknown source class: {source_class}")
            target_class = (mapping.get("target_class") or "").strip()
            enabled = bool(mapping.get("enabled", True))
            action = mapping.get("action")
            target_exists = mapping.get("target_exists")
            if enabled:
                self.dataset_service._validate_class_name(target_class)
                enabled_count += 1
            mapping_by_source[source_class] = {
                "source_class": source_class,
                "target_class": target_class,
                "enabled": enabled,
                "action": action,
                "target_exists": target_exists,
            }

        if enabled_count == 0:
            raise HTTPException(status_code=400, detail="Enable at least one class mapping before merging.")

        merge_id = datetime.now(timezone.utc).strftime("merge_%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        source_tag = self._derive_source_tag(source_root)

        copied_count = 0
        skipped_existing = 0
        skipped_disabled = 0
        per_target_counts: Dict[str, Dict[str, int]] = {}
        copied_records = []

        for source_entry in source_scan["classes"]:
            source_class = source_entry["name"]
            mapping = mapping_by_source.get(source_class)
            if not mapping or not mapping["enabled"] or not mapping["target_class"]:
                skipped_disabled += source_entry["total"]
                continue

            target_class = mapping["target_class"]
            files_by_split = self.dataset_service.list_class_images(str(source_root), source_class)
            for split, files in files_by_split.items():
                if split not in KNOWN_SPLITS:
                    continue
                target_dir = target_root / split / target_class
                target_dir.mkdir(parents=True, exist_ok=True)
                split_counter = per_target_counts.setdefault(target_class, {name: 0 for name in KNOWN_SPLITS})

                for source_file in files:
                    target_name = self._build_target_name(source_tag, split, source_class, source_file.name)
                    target_file = target_dir / target_name
                    if target_file.exists():
                        if filecmp.cmp(source_file, target_file, shallow=False):
                            skipped_existing += 1
                            continue
                        target_file = self.dataset_service._build_unique_path(target_file)

                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    split_counter[split] += 1
                    copied_records.append(
                        {
                            "source": str(source_file.relative_to(source_root)),
                            "target": str(target_file.relative_to(target_root)),
                        }
                    )

        merge_record = {
            "merge_id": merge_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source_root),
            "target_path": str(target_root),
            "source_tag": source_tag,
            "copied_count": copied_count,
            "skipped_existing": skipped_existing,
            "skipped_disabled": skipped_disabled,
            "class_mappings": list(mapping_by_source.values()),
            "per_target_counts": per_target_counts,
            "records": copied_records,
        }
        self._write_merge_record(target_root, merge_id, merge_record)

        return {
            "merge_id": merge_id,
            "copied_count": copied_count,
            "skipped_existing": skipped_existing,
            "skipped_disabled": skipped_disabled,
            "target_path": str(target_root),
            "per_target_counts": per_target_counts,
        }

    def read_merge_history(self, raw_dataset_path: str, limit: int = 10) -> dict:
        dataset_root = self.dataset_service.ensure_dataset_root(raw_dataset_path)
        merge_dir = dataset_root / ".dataset_studio" / "merges"
        if not merge_dir.is_dir():
            return {"entries": []}

        files = sorted((path for path in merge_dir.glob("*.json") if path.is_file()), reverse=True)
        entries = []
        for path in files[:limit]:
            try:
                entries.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return {"entries": entries}

    def rename_class_references(self, raw_dataset_path: str, old_name: str, new_name: str) -> dict:
        dataset_root = self.dataset_service.ensure_dataset_root(raw_dataset_path)
        merge_dir = dataset_root / ".dataset_studio" / "merges"
        if not merge_dir.is_dir():
            return {
                "updated_files": 0,
                "updated_mappings": 0,
                "updated_target_counts": 0,
                "updated_records": 0,
            }

        updated_files = 0
        updated_mappings = 0
        updated_target_counts = 0
        updated_records = 0

        for path in sorted(merge_dir.glob("*.json")):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            changed = False

            for mapping in payload.get("class_mappings", []):
                if mapping.get("target_class") == old_name:
                    mapping["target_class"] = new_name
                    updated_mappings += 1
                    changed = True

            per_target_counts = payload.get("per_target_counts")
            if isinstance(per_target_counts, dict) and old_name in per_target_counts:
                old_counts = per_target_counts.pop(old_name)
                merged_counts = per_target_counts.setdefault(
                    new_name,
                    {split: 0 for split in KNOWN_SPLITS},
                )
                for split in KNOWN_SPLITS:
                    merged_counts[split] = int(merged_counts.get(split, 0)) + int(old_counts.get(split, 0))
                updated_target_counts += 1
                changed = True

            for record in payload.get("records", []):
                target_value = record.get("target")
                if not isinstance(target_value, str):
                    continue
                target_path = PurePosixPath(target_value)
                parts = list(target_path.parts)
                if len(parts) >= 3 and parts[1] == old_name:
                    parts[1] = new_name
                    record["target"] = str(PurePosixPath(*parts))
                    updated_records += 1
                    changed = True

            if changed:
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                updated_files += 1

        return {
            "updated_files": updated_files,
            "updated_mappings": updated_mappings,
            "updated_target_counts": updated_target_counts,
            "updated_records": updated_records,
        }

    def _scan_target_for_preview(self, target_root: Path) -> dict:
        if target_root.exists():
            return self.dataset_service.scan_dataset(str(target_root))
        return {
            "path": str(target_root),
            "relative_path": str(target_root.relative_to(DATA_ROOT)),
            "kind": "working_dataset",
            "class_count": 0,
            "total_images": 0,
            "splits": {},
            "classes": [],
            "missing": True,
        }

    def _derive_source_tag(self, source_root: Path) -> str:
        relative = source_root.relative_to(DATA_ROOT)
        if relative.parts:
            base = relative.parts[0]
        else:
            base = source_root.name
        return self.dataset_service._validate_class_name(base.replace(" ", "_"))

    def _build_target_name(self, source_tag: str, split: str, source_class: str, filename: str) -> str:
        return f"{source_tag}__{split}__{source_class}__{filename}"

    def _write_merge_record(self, dataset_root: Path, merge_id: str, payload: dict) -> None:
        merge_dir = dataset_root / ".dataset_studio" / "merges"
        merge_dir.mkdir(parents=True, exist_ok=True)
        record_path = merge_dir / f"{merge_id}.json"
        record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
