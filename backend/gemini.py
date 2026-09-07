"""Google Gemini REST API client for DrakeAI."""

import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from backend.config import settings

logger = logging.getLogger("drakeai.gemini")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _build_payload(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
    json_mode: bool = False
) -> Dict[str, Any]:
    """Formats payload for Gemini v1beta generateContent API."""
    contents: List[Dict[str, Any]] = []

    if history:
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("content", "").strip()
            if text:
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })

    # Add the current user query
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        }
    }

    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    if system_instruction and system_instruction.strip():
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction.strip()}]
        }

    return payload


def _extract_response_text(data: Dict[str, Any]) -> str:
    """Extracts synthesized text from Gemini API response candidates."""
    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        block_reason = feedback.get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini generation blocked: {block_reason}")
        raise RuntimeError(f"Gemini returned no candidates: {data}")

    first_candidate = candidates[0]
    content = first_candidate.get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise RuntimeError("Gemini candidate content has no parts.")

    # Combine all parts if multiple
    texts = [p.get("text", "") for p in parts if "text" in p]
    return "".join(texts).strip()


def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: float = 30.0
) -> str:
    """
    Synchronously calls Google Gemini generateContent API.
    Raises RuntimeError on failure or if API key is not configured.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in settings or environment.")

    model_name = (model or settings.GEMINI_MODEL or "gemini-flash-latest").strip()
    if not model_name or "2.5" in model_name:
        model_name = "gemini-flash-latest"
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = _build_payload(
        prompt=prompt,
        system_instruction=system_instruction,
        history=history,
        temperature=temperature,
        json_mode=json_mode
    )

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            error_msg = response.text
            try:
                err_json = response.json()
                error_msg = err_json.get("error", {}).get("message", response.text)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API error ({response.status_code}): {error_msg}")

        data = response.json()
        return _extract_response_text(data)


async def acall_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: float = 30.0
) -> str:
    """
    Asynchronously calls Google Gemini generateContent API.
    Raises RuntimeError on failure or if API key is not configured.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in settings or environment.")

    model_name = (model or settings.GEMINI_MODEL or "gemini-flash-latest").strip()
    if not model_name or "2.5" in model_name:
        model_name = "gemini-flash-latest"
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = _build_payload(
        prompt=prompt,
        system_instruction=system_instruction,
        history=history,
        temperature=temperature,
        json_mode=json_mode
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            error_msg = response.text
            try:
                err_json = response.json()
                error_msg = err_json.get("error", {}).get("message", response.text)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API error ({response.status_code}): {error_msg}")

        data = response.json()
        return _extract_response_text(data)
