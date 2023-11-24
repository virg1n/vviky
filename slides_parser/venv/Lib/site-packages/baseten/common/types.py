from dataclasses import dataclass
from typing import Optional


@dataclass
class TryPatchDraftTrussResponse:
    succeeded: bool
    needs_full_deploy: bool
    model_version_info: Optional[dict] = None
    error_message: Optional[str] = None
    success_message: Optional[str] = None

    @staticmethod
    def needs_full_deploy_response(
        error_message: Optional[str] = None,
    ) -> "TryPatchDraftTrussResponse":
        return TryPatchDraftTrussResponse(
            succeeded=False,
            needs_full_deploy=True,
            error_message=error_message,
        )

    @staticmethod
    def success_response(model_version_info: dict, success_message: Optional[str] = None):
        return TryPatchDraftTrussResponse(
            succeeded=True,
            needs_full_deploy=False,
            model_version_info=model_version_info,
            success_message=success_message,
        )
