import json
from typing import Any, Dict, List

import requests

from api import API_BASE_URL


class APIClient:
    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def ask_question(self, question: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/ask",
            json={"question": question},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def upload_files(self, files: List[Any]) -> Dict[str, Any]:
        payload = []
        for uploaded_file in files:
            payload.append(
                (
                    "files",
                    (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    ),
                )
            )

        response = requests.post(
            f"{self.base_url}/upload",
            files=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
