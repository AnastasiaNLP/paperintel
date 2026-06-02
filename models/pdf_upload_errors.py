class PdfUploadNotFoundError(ValueError):
    def __init__(self, upload_id: str) -> None:
        super().__init__(f"PDF upload not found: {upload_id}")
        self.upload_id = upload_id


class PdfUploadExpiredError(ValueError):
    def __init__(self, upload_id: str) -> None:
        super().__init__(f"PDF upload has expired: {upload_id}")
        self.upload_id = upload_id


class PdfUploadStateError(ValueError):
    def __init__(self, *, upload_id: str, status: str, target_status: str) -> None:
        super().__init__(
            f"Cannot transition PDF upload {upload_id} from {status} "
            f"to {target_status}."
        )
        self.upload_id = upload_id
        self.status = status
        self.target_status = target_status


class PdfUploadChecksumMismatchError(ValueError):
    pass


class PdfUploadInvalidContentError(ValueError):
    pass


class PdfUploadSizeMismatchError(ValueError):
    pass
