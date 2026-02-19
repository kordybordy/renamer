import glob
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Dict

from PIL import Image
import pytesseract

from app_logging import log_info
from app_runtime import BASE_DIR, PDFTOPPM_EXE


TEXT_LAYER_MIN_CHARS = 400


def configure_tesseract() -> str:
    tesseract_dir = os.path.join(BASE_DIR, "tesseract")
    tessdata_dir = os.path.join(tesseract_dir, "tessdata")
    tesseract_exe = os.path.join(tesseract_dir, "tesseract.exe")

    if not os.path.exists(tesseract_exe):
        raise RuntimeError(f"Tesseract EXE not found: {tesseract_exe}")

    if not os.path.exists(tessdata_dir):
        raise RuntimeError(f"Tessdata folder not found: {tessdata_dir}")

    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return tessdata_dir


def extract_text_ocr(pdf_path: str, char_limit: int, dpi: int, pages: int) -> str:
    temp_dir = tempfile.mkdtemp(prefix="ocr_")
    try:
        cmd = [
            PDFTOPPM_EXE,
            "-png",
            "-f",
            "1",
            "-l",
            str(max(1, pages)),
            "-r",
            str(dpi),
            pdf_path,
            os.path.join(temp_dir, "page"),
        ]
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        image_paths = sorted(glob.glob(os.path.join(temp_dir, "page-*.png")))
        if not image_paths:
            image_paths = sorted(glob.glob(os.path.join(temp_dir, "page-*.ppm")))

        if not image_paths:
            raise RuntimeError(
                f"pdftoppm produced no output images for '{os.path.basename(pdf_path)}'"
            )

        text_chunks: list[str] = []
        for image_file in image_paths:
            chunk = pytesseract.image_to_string(Image.open(image_file), lang="pol")
            text_chunks.append(chunk)
            if len("".join(text_chunks)) >= char_limit:
                break
        text = "".join(text_chunks)[:char_limit]
        return text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_text_layer(pdf_path: str, char_limit: int) -> str:
    pdftotext_exe = os.path.join(os.path.dirname(PDFTOPPM_EXE), "pdftotext.exe")
    pdftotext_cmd = pdftotext_exe if os.path.exists(pdftotext_exe) else "pdftotext"
    cmd = [
        pdftotext_cmd,
        "-enc",
        "UTF-8",
        "-f",
        "1",
        "-l",
        "3",
        pdf_path,
        "-",
    ]
    kwargs = {
        "check": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run(cmd, **kwargs)
        return (result.stdout or "")[:char_limit].strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log_info(
            f"Text-layer extraction unavailable for '{os.path.basename(pdf_path)}': {exc}"
        )
        return ""


OCR_CACHE: Dict[str, dict] = {}
OCR_CACHE_LOCK = threading.Lock()


def get_ocr_text(pdf_path: str, char_limit: int, dpi: int, pages: int) -> str:
    normalized_pages = max(1, pages)
    with OCR_CACHE_LOCK:
        cached = OCR_CACHE.get(pdf_path)
        if (
            cached
            and cached.get("dpi") == dpi
            and cached.get("pages") == normalized_pages
            and cached.get("char_limit", 0) >= char_limit
        ):
            cached_text = cached.get("ocr_text", "")[:char_limit]
            log_info(
                f"Reusing cached OCR for '{os.path.basename(pdf_path)}' "
                f"(pages={normalized_pages}, dpi={dpi}, char_limit={char_limit})"
            )
            return cached_text

    text_layer = extract_text_layer(pdf_path, char_limit)
    if len(text_layer) >= TEXT_LAYER_MIN_CHARS:
        log_info(
            f"Using text-layer extraction for '{os.path.basename(pdf_path)}' "
            f"({len(text_layer)} chars, OCR skipped)"
        )
        with OCR_CACHE_LOCK:
            OCR_CACHE[pdf_path] = {
                "ocr_text": text_layer,
                "char_limit": max(char_limit, len(text_layer)),
                "dpi": dpi,
                "pages": normalized_pages,
            }
        return text_layer

    text = extract_text_ocr(pdf_path, char_limit, dpi, normalized_pages)
    with OCR_CACHE_LOCK:
        OCR_CACHE[pdf_path] = {
            "ocr_text": text,
            "char_limit": max(char_limit, len(text)),
            "dpi": dpi,
            "pages": normalized_pages,
        }
    return text


TESSDATA_DIR = configure_tesseract()
