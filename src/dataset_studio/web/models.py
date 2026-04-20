from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


class MergePreviewRequest(BaseModel):
    source_path: str = Field(..., min_length=1)
    target_path: str = Field(..., min_length=1)


class MergeMapping(BaseModel):
    source_class: str = Field(..., min_length=1)
    target_class: Optional[str] = None
    enabled: bool = True
    action: Optional[str] = None
    target_exists: Optional[bool] = None


class MergeCommitRequest(BaseModel):
    source_path: str = Field(..., min_length=1)
    target_path: str = Field(..., min_length=1)
    mappings: List[MergeMapping]


class ClassDetailRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    class_name: str = Field(..., min_length=1)
    limit_per_split: int = Field(default=120, ge=1, le=2000)


class RenameClassRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    old_name: str = Field(..., min_length=1)
    new_name: str = Field(..., min_length=1)


class ReassignImageRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    target_class: str = Field(..., min_length=1)
    target_split: str = Field(..., min_length=1)


class TrashImageRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)


class BulkReassignImagesRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    image_paths: List[str] = Field(..., min_length=1)
    target_class: str = Field(..., min_length=1)
    target_split: str = Field(..., min_length=1)


class BulkTrashImagesRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    image_paths: List[str] = Field(..., min_length=1)


class NpzExportRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)
    output_filename: str = Field(default="")
    per_class_limit: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=32, ge=1, le=256)
    selected_paths_by_class: Dict[str, List[str]] = Field(default_factory=dict)


class SelectionManifestRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)
    output_filename: str = Field(default="")
    per_class_limit: int = Field(default=20, ge=1, le=200)
    selected_paths_by_class: Dict[str, List[str]] = Field(default_factory=dict)


class SessionStateRequest(BaseModel):
    target_path: str = Field(default="")
    source_path: str = Field(default="")
    preview: Optional[dict[str, Any]] = None
    target_scan: Optional[dict[str, Any]] = None
    selected_class: Optional[str] = None
    class_detail_split: Optional[str] = None
    current_page: str = Field(default="merge")
    browser_filter: str = Field(default="all")
    browser_sort: str = Field(default="size_desc")
    merge_filter: str = Field(default="all")
    class_search: str = Field(default="")
    starred_classes: List[str] = Field(default_factory=list)
    export_model_path: str = Field(default="")
    export_output_filename: str = Field(default="")
    export_selected_class: Optional[str] = None
    export_search: str = Field(default="")
    export_filter: str = Field(default="all")
    export_sort: str = Field(default="size_desc")
    export_detail_mode: str = Field(default="all")
    export_per_class_limit: int = Field(default=20)
    export_selections: Dict[str, List[str]] = Field(default_factory=dict)
    export_status: Optional[dict[str, Any]] = None
    export_result: Optional[dict[str, Any]] = None
