from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from dataset_studio.web.deps import dataset_service, embedding_export_service, merge_service, session_service
from dataset_studio.web.models import (
    BulkReassignImagesRequest,
    BulkTrashImagesRequest,
    ClassDetailRequest,
    DatasetPathRequest,
    MergeCommitRequest,
    MergePreviewRequest,
    NpzExportRequest,
    ReassignImageRequest,
    RenameClassRequest,
    SelectionManifestRequest,
    SessionStateRequest,
    TrashImageRequest,
)

router = APIRouter(prefix="/api")


def _resolved_export_filename(model_path: str, output_filename: str, per_class_limit: int) -> str:
    candidate = (output_filename or "").strip()
    if not candidate:
        candidate = f"{Path(model_path).stem}_gallery_{per_class_limit}.npz"
    if not candidate.lower().endswith(".npz"):
        candidate = f"{candidate}.npz"
    return Path(candidate).name


def _scan_dataset_with_history(path: str) -> dict:
    scan = dataset_service.scan_dataset(path)
    scan["merge_history"] = merge_service.read_merge_history(path)["entries"]
    return scan


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/datasets/discover")
def discover_datasets() -> dict:
    items = dataset_service.discover_datasets()
    return {
        "items": items,
        "working_datasets": [item for item in items if item["kind"] == "working_dataset"],
        "recognition_exports": [item for item in items if item["kind"] == "recognition_export"],
    }


@router.post("/datasets/scan")
def scan_dataset(request: DatasetPathRequest) -> dict:
    return _scan_dataset_with_history(request.path)


@router.get("/session/current")
def current_session() -> dict:
    return {"session": session_service.get_or_create_current_session()}


@router.post("/session/current/state")
def save_current_session_state(request: SessionStateRequest) -> dict:
    session = session_service.save_current_session_state(request.model_dump())
    return {
        "session_id": session["session_id"],
        "updated_at": session["updated_at"],
    }


@router.post("/merge/preview")
def merge_preview(request: MergePreviewRequest) -> dict:
    return merge_service.build_preview(request.target_path, request.source_path)


@router.post("/merge/commit")
def merge_commit(request: MergeCommitRequest) -> dict:
    payload = merge_service.commit_merge(
        target_path=request.target_path,
        source_path=request.source_path,
        mappings=[mapping.model_dump() for mapping in request.mappings],
    )
    payload["target_scan"] = _scan_dataset_with_history(request.target_path)
    payload["merge_history"] = payload["target_scan"]["merge_history"]
    session_service.record_merge(
        source_path=request.source_path,
        target_path=request.target_path,
        merge_payload=payload,
        preview=session_service.get_or_create_current_session().get("preview"),
    )
    return payload


@router.post("/classes/detail")
def class_detail(request: ClassDetailRequest) -> dict:
    return dataset_service.get_class_detail(
        raw_dataset_path=request.dataset_path,
        class_name=request.class_name,
        limit_per_split=request.limit_per_split,
    )


@router.post("/classes/rename")
def rename_class(request: RenameClassRequest) -> dict:
    payload = dataset_service.rename_class(
        raw_dataset_path=request.dataset_path,
        old_name=request.old_name,
        new_name=request.new_name,
    )
    if payload.get("renamed"):
        payload["merge_history_updates"] = merge_service.rename_class_references(
            raw_dataset_path=request.dataset_path,
            old_name=request.old_name,
            new_name=request.new_name,
        )
    payload["dataset"] = _scan_dataset_with_history(request.dataset_path)
    return payload


@router.post("/images/reassign")
def reassign_image(request: ReassignImageRequest) -> dict:
    payload = dataset_service.reassign_image(
        raw_dataset_path=request.dataset_path,
        raw_image_path=request.image_path,
        target_class=request.target_class,
        target_split=request.target_split,
    )
    payload["dataset"] = _scan_dataset_with_history(request.dataset_path)
    return payload


@router.post("/images/reassign-many")
def reassign_images(request: BulkReassignImagesRequest) -> dict:
    payload = dataset_service.reassign_images(
        raw_dataset_path=request.dataset_path,
        raw_image_paths=request.image_paths,
        target_class=request.target_class,
        target_split=request.target_split,
    )
    payload["dataset"] = _scan_dataset_with_history(request.dataset_path)
    return payload


@router.post("/images/trash")
def trash_image(request: TrashImageRequest) -> dict:
    payload = dataset_service.trash_image(
        raw_dataset_path=request.dataset_path,
        raw_image_path=request.image_path,
    )
    payload["dataset"] = _scan_dataset_with_history(request.dataset_path)
    return payload


@router.post("/images/trash-many")
def trash_images(request: BulkTrashImagesRequest) -> dict:
    payload = dataset_service.trash_images(
        raw_dataset_path=request.dataset_path,
        raw_image_paths=request.image_paths,
    )
    payload["dataset"] = _scan_dataset_with_history(request.dataset_path)
    return payload


@router.post("/export/npz")
def export_npz(request: NpzExportRequest) -> dict:
    resolved_filename = _resolved_export_filename(
        model_path=request.model_path,
        output_filename=request.output_filename,
        per_class_limit=request.per_class_limit,
    )
    session_service.update_export_state(
        status={
            "tone": "running",
            "title": "Exporting NPZ",
            "message": "Embedding the selected train images and writing the gallery into the dataset db folder.",
            "detail": str(Path(request.dataset_path) / "db" / resolved_filename),
        },
        result=None,
    )
    try:
        result = embedding_export_service.export_dataset_npz(
            raw_dataset_path=request.dataset_path,
            raw_model_path=request.model_path,
            raw_output_filename=request.output_filename,
            per_class_limit=request.per_class_limit,
            batch_size=request.batch_size,
            selected_paths_by_class=request.selected_paths_by_class,
        )
    except HTTPException as exc:
        session_service.update_export_state(
            status={
                "tone": "error",
                "title": "NPZ export failed",
                "message": "Dataset Studio hit an error while generating the gallery file.",
                "detail": str(exc.detail),
            },
            result=None,
        )
        raise
    except Exception:
        session_service.update_export_state(
            status={
                "tone": "error",
                "title": "NPZ export failed",
                "message": "Dataset Studio hit an unexpected error while generating the gallery file.",
                "detail": str(Path(request.dataset_path) / "db" / resolved_filename),
            },
            result=None,
        )
        raise

    session_service.update_export_state(
        status={
            "tone": "success",
            "title": "NPZ export complete",
            "message": f"{result['image_count']} images across {result['class_count']} classes were embedded.",
            "detail": f"{result['output_path']}\nSelection: {result['selection_manifest_path']}",
        },
        result=result,
    )
    return result


@router.post("/export/selection/save")
def save_export_selection(request: SelectionManifestRequest) -> dict:
    return embedding_export_service.save_selection_manifest(
        raw_dataset_path=request.dataset_path,
        raw_model_path=request.model_path,
        raw_output_filename=request.output_filename,
        per_class_limit=request.per_class_limit,
        selected_paths_by_class=request.selected_paths_by_class,
    )


@router.get("/images/file")
def image_file(path: str) -> FileResponse:
    image_path = dataset_service.resolve_data_path(path)
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    return FileResponse(image_path)
