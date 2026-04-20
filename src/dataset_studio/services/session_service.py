from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataset_studio.services.dataset_service import DatasetService
from dataset_studio.services.merge_service import MergeService


DEFAULT_TARGET_PATH = "/workspace/vdata/dessert_poc/recognition_dataset_combined_711_whiskey"
DEFAULT_SOURCE_PATH = "/workspace/vdata/victor/runs/run_20260415_97e0/outputs/recognition_dataset/sort_20260415_a7ca"
DEFAULT_EXPORT_MODEL_PATH = "/workspace/anand/models/recognition_models/checkpoints/dinov3b_dessert_v1/best_dinov3b_dessert_v1_classifier.pth"
DEFAULT_EXPORT_PER_CLASS_LIMIT = 20
_UNSET = object()


class SessionService:
    def __init__(self, dataset_service: DatasetService, merge_service: MergeService) -> None:
        self.dataset_service = dataset_service
        self.merge_service = merge_service
        self.app_root = Path(__file__).resolve().parents[3]
        self.session_root = self.app_root / ".sessions"
        self.current_pointer = self.session_root / "current_session.json"
        self.session_root.mkdir(parents=True, exist_ok=True)

    def get_or_create_current_session(self) -> dict[str, Any]:
        return self.hydrate_session(self._get_or_create_current_session_raw())

    def save_current_session_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._get_or_create_current_session_raw()
        session["target_path"] = payload.get("target_path") or session.get("target_path") or DEFAULT_TARGET_PATH
        session["source_path"] = payload.get("source_path") or session.get("source_path") or DEFAULT_SOURCE_PATH
        session["preview"] = payload.get("preview")
        session["target_scan"] = payload.get("target_scan")
        session["selected_class"] = payload.get("selected_class")
        session["class_detail_split"] = payload.get("class_detail_split")
        session["current_page"] = payload.get("current_page") or "merge"
        session["browser_filter"] = payload.get("browser_filter") or "all"
        session["browser_sort"] = payload.get("browser_sort") or "size_desc"
        session["merge_filter"] = payload.get("merge_filter") or "all"
        session["class_search"] = payload.get("class_search") or ""
        session["starred_classes"] = payload.get("starred_classes") or []
        session["export_model_path"] = payload.get("export_model_path") or session.get("export_model_path") or DEFAULT_EXPORT_MODEL_PATH
        session["export_selected_class"] = payload.get("export_selected_class")
        session["export_search"] = payload.get("export_search") or ""
        session["export_filter"] = payload.get("export_filter") or "all"
        session["export_sort"] = payload.get("export_sort") or "size_desc"
        session["export_detail_mode"] = payload.get("export_detail_mode") or "all"
        session["export_per_class_limit"] = int(payload.get("export_per_class_limit") or session.get("export_per_class_limit") or DEFAULT_EXPORT_PER_CLASS_LIMIT)
        session["export_selections"] = payload.get("export_selections") or {}
        session["export_status"] = payload.get("export_status") if payload.get("export_status") is not None else session.get("export_status") or self._default_export_status()
        session["export_result"] = payload.get("export_result")
        return self._write_session(session, set_current=True, hydrate_result=False)

    def update_export_state(
        self,
        *,
        status: dict[str, Any] | None = None,
        result: dict[str, Any] | None | object = _UNSET,
    ) -> dict[str, Any]:
        session = self._get_or_create_current_session_raw()
        if status is not None:
            session["export_status"] = status
        if result is not _UNSET:
            session["export_result"] = result
        return self._write_session(session, set_current=True, hydrate_result=False)

    def record_merge(
        self,
        *,
        source_path: str,
        target_path: str,
        merge_payload: dict[str, Any],
        preview: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session = self._get_or_create_current_session_raw()
        session["target_path"] = target_path
        session["source_path"] = source_path
        session["preview"] = preview
        session.setdefault("merge_events", [])
        session["merge_events"].append(
            {
                "merge_id": merge_payload.get("merge_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_path": source_path,
                "target_path": target_path,
                "copied_count": merge_payload.get("copied_count", 0),
                "skipped_existing": merge_payload.get("skipped_existing", 0),
                "skipped_disabled": merge_payload.get("skipped_disabled", 0),
                "per_target_counts": merge_payload.get("per_target_counts", {}),
            }
        )
        session["target_scan"] = merge_payload.get("target_scan")
        return self._write_session(session, set_current=True, hydrate_result=False)

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        path = self.session_root / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def hydrate_session(self, session: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(session)
        target_path = hydrated.get("target_path")
        source_path = hydrated.get("source_path")
        hydrated["merge_events"] = hydrated.get("merge_events", [])

        if target_path:
            try:
                target_scan = self.dataset_service.scan_dataset(target_path)
                target_scan["merge_history"] = self.merge_service.read_merge_history(target_path)["entries"]
                hydrated["target_scan"] = target_scan
            except Exception:
                hydrated["target_scan"] = hydrated.get("target_scan")

        if hydrated.get("preview"):
            preview = dict(hydrated["preview"])
            if source_path:
                try:
                    preview["source"] = self.dataset_service.scan_dataset(source_path)
                except Exception:
                    pass
            if hydrated.get("target_scan"):
                preview["target"] = hydrated["target_scan"]
            hydrated["preview"] = preview

        return hydrated

    def _new_session_payload(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        session_id = datetime.now(timezone.utc).strftime("sess_%Y%m%d_%H%M%S_") + uuid4().hex[:6]
        return {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "target_path": DEFAULT_TARGET_PATH,
            "source_path": DEFAULT_SOURCE_PATH,
            "preview": None,
            "target_scan": None,
            "selected_class": None,
            "class_detail_split": None,
            "current_page": "merge",
            "browser_filter": "all",
            "browser_sort": "size_desc",
            "merge_filter": "all",
            "class_search": "",
            "starred_classes": [],
            "export_model_path": DEFAULT_EXPORT_MODEL_PATH,
            "export_selected_class": None,
            "export_search": "",
            "export_filter": "all",
            "export_sort": "size_desc",
            "export_detail_mode": "all",
            "export_per_class_limit": DEFAULT_EXPORT_PER_CLASS_LIMIT,
            "export_selections": {},
            "export_status": self._default_export_status(),
            "export_result": None,
            "merge_events": [],
        }

    @staticmethod
    def _default_export_status() -> dict[str, Any]:
        return {
            "tone": "idle",
            "title": "No export activity yet",
            "message": "Save a selection snapshot or run an export to see progress and results here.",
            "detail": "",
        }

    def _get_or_create_current_session_raw(self) -> dict[str, Any]:
        current_id = self._read_current_session_id()
        if current_id:
            session = self.load_session(current_id)
            if session:
                return session
        session = self._new_session_payload()
        return self._write_session(session, set_current=True, hydrate_result=False)

    def _write_session(self, session: dict[str, Any], *, set_current: bool, hydrate_result: bool) -> dict[str, Any]:
        session = dict(session)
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.session_root.mkdir(parents=True, exist_ok=True)
        path = self.session_root / f"{session['session_id']}.json"
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        if set_current:
            self.current_pointer.write_text(json.dumps({"session_id": session["session_id"]}, indent=2), encoding="utf-8")
        if hydrate_result:
            return self.hydrate_session(session)
        return session

    def _read_current_session_id(self) -> str | None:
        if not self.current_pointer.is_file():
            return None
        try:
            payload = json.loads(self.current_pointer.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        session_id = payload.get("session_id")
        if not session_id:
            return None
        return str(session_id)
