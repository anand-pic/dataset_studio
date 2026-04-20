from dataset_studio.services.dataset_service import DatasetService
from dataset_studio.services.embedding_export_service import EmbeddingExportService
from dataset_studio.services.merge_service import MergeService
from dataset_studio.services.session_service import SessionService

dataset_service = DatasetService()
embedding_export_service = EmbeddingExportService(dataset_service)
merge_service = MergeService(dataset_service)
session_service = SessionService(dataset_service, merge_service)
