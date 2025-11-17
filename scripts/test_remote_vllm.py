#!/usr/bin/env python
import argparse
import base64
import json
from pathlib import Path
from typing import Iterable, Optional, Any
import sys

import requests

DEFAULT_PROMPT = (
    "Please perform high-fidelity OCR on this page and return markdown with layout."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Test remote Chandra vLLM service via OpenAI-compatible /v1/chat/completions API."
        )
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/v1",
        help="Base URL of the remote vLLM server (default: http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="Bearer token expected by the remote server (default: EMPTY).",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        dest="image_paths",
        help="Path to a document image. Use multiple times for batch processing.",
    )
    parser.add_argument(
        "--input-dir",
        help="Process all images in this directory (png/jpg/jpeg/webp/tiff/bmp).",
    )
    parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob pattern when --input-dir is set (default: *.png).",
    )
    parser.add_argument(
        "--model",
        default="chandra",
        help="Model name served by the remote vLLM instance (default: chandra).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Custom instruction to send with each request.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a text file containing the prompt (overrides --prompt).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate from the OCR response (default: 2048).",
    )
    parser.add_argument(
        "--output-dir",
        default="remote_results",
        help="Directory to store output JSON/Markdown (default: ./remote_results).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop processing on first failure.",
    )
    return parser


def encode_image(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def call_remote_ocr(
    endpoint: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
) -> dict:
    image_b64 = encode_image(image_path)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def extract_markdown(response_json: dict) -> Optional[str]:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    return text
    return None


def iter_images(
    paths: Optional[Iterable[str]], input_dir: Optional[str], pattern: str
) -> list[Path]:
    images: list[Path] = []
    if paths:
        images.extend(Path(p).expanduser().resolve() for p in paths)
    if input_dir:
        root = Path(input_dir).expanduser().resolve()
        images.extend(sorted(root.glob(pattern)))
    if not images:
        default = Path(__file__).resolve().parents[1] / "assets/examples/math/attn_all.png"
        images.append(default)
    deduped = []
    seen = set()
    for img in images:
        if img.exists() and img not in seen:
            deduped.append(img)
            seen.add(img)
    return deduped


def save_outputs(
    response_json: dict,
    markdown: Optional[str],
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(json.dumps(response_json, indent=2), encoding="utf-8")
    if markdown:
        (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")


def main():
    parser = build_parser()
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")
    endpoint = f"{server_url}/chat/completions"

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = args.prompt

    images = iter_images(args.image_paths, args.input_dir, args.glob)
    if not images:
        print("No images found to process.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    summary = []
    for image_path in images:
        try:
            response = call_remote_ocr(
                endpoint,
                api_key=args.api_key,
                model=args.model,
                image_path=image_path,
                prompt=prompt,
                max_tokens=args.max_output_tokens,
            )
            markdown = extract_markdown(response)
            stem = image_path.stem
            save_outputs(response, markdown, output_dir, stem)
            preview = (markdown or "")[:2000]
            print(f"\n[{stem}] OCR preview:\n{preview}\n")
            summary.append((image_path, "success", None))
        except Exception as exc:
            message = str(exc)
            summary.append((image_path, "failed", message))
            print(f"[{image_path}] failed: {message}", file=sys.stderr)
            if args.fail_fast:
                break

    print("\n=== Batch Summary ===")
    for image_path, status, msg in summary:
        if status == "success":
            print(f"{image_path}: success")
        else:
            print(f"{image_path}: failed - {msg}")


if __name__ == "__main__":
    main()

