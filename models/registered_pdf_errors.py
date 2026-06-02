class RegisteredPdfBlobNotFoundError(ValueError):
    def __init__(self, blob_id: str) -> None:
        super().__init__(f"Registered PDF blob was not found: {blob_id}")
        self.blob_id = blob_id


class RegisteredPdfBlobNotAuthorizedError(ValueError):
    def __init__(self, *, session_id: str, blob_id: str) -> None:
        super().__init__(
            f"Session {session_id} is not authorized to use registered PDF blob: {blob_id}"
        )
        self.session_id = session_id
        self.blob_id = blob_id
