import io
import os
import re
import ast
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

PROMPT_PERFIX='<image>\n<|grounding|>'
PROMPT = 'Convert the document to markdown.'
CROP_MODE = True

# Register custom model class with vLLM
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def _init_llm(model_path: str, max_concurrency: int) -> LLM:
    llm_local = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=64,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_concurrency,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )
    return llm_local


# Global singleton LLM and its current settings
llm: Optional[LLM] = None
CURRENT_MODEL_PATH: Optional[str] = None
CURRENT_MAX_CONCURRENCY: Optional[int] = None
import threading
_llm_lock = threading.Lock()


def _ensure_llm(model_path: Optional[str], max_concurrency: Optional[int]) -> None:
    """Ensure a global LLM exists with the requested settings.

    If any of the parameters differ from the current ones, reinitialize.
    Fallback to environment if not provided.
    """
    global llm, CURRENT_MODEL_PATH, CURRENT_MAX_CONCURRENCY

    # Defaults from environment
    env_model = os.environ.get("DS_MODEL_PATH")
    env_conc = os.environ.get("DS_MAX_CONCURRENCY")

    target_model = model_path or CURRENT_MODEL_PATH or env_model
    if target_model is None:
        raise RuntimeError("Model path not provided. Set DS_MODEL_PATH or pass model_path.")

    if max_concurrency is None:
        if CURRENT_MAX_CONCURRENCY is not None:
            target_conc = CURRENT_MAX_CONCURRENCY
        elif env_conc is not None:
            try:
                target_conc = int(env_conc)
            except Exception:
                raise RuntimeError("Invalid DS_MAX_CONCURRENCY; must be int")
        else:
            target_conc = 32  # sensible default
    else:
        target_conc = int(max_concurrency)
    if target_conc < 1:
        raise RuntimeError("max_concurrency must be >= 1")

    with _llm_lock:
        need_reinit = (
            llm is None
            or CURRENT_MODEL_PATH != target_model
            or CURRENT_MAX_CONCURRENCY != target_conc
        )
        if need_reinit:
            llm = _init_llm(target_model, target_conc)
            CURRENT_MODEL_PATH = target_model
            CURRENT_MAX_CONCURRENCY = target_conc


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=[
        NoRepeatNGramLogitsProcessor(
            ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
        )
    ],
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


def _pdf_bytes_to_images_high_quality(pdf_bytes: bytes, dpi: int = 144) -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    pdf_document.close()
    return images


def _load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image.convert("RGB")


REF_PATTERN = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"

def _clean_text_and_collect_regions(
    raw_text: str, image_w: int, image_h: int, page_index: int
) -> Dict[str, Any]:
    # Remove trailing special EOS if present
    text = raw_text.replace("<｜end▁of▁sentence｜>", "")

    matches = re.findall(REF_PATTERN, text, re.DOTALL)
    image_matches = []
    other_matches = []
    for m in matches:
        if "<|ref|>image<|/ref|>" in m[0]:
            image_matches.append(m[0])
        else:
            other_matches.append(m[0])

    # Extract regions in pixel coords
    regions: List[Dict[str, Any]] = []
    for full, label, coords_str in matches:
        try:
            coords = ast.literal_eval(coords_str)
        except Exception:
            continue
        if not isinstance(coords, (list, tuple)):
            continue
        for c in coords:
            try:
                x1, y1, x2, y2 = c
                # Inputs normalized by 999 per codebase convention
                px1 = int(x1 / 999 * image_w)
                py1 = int(y1 / 999 * image_h)
                px2 = int(x2 / 999 * image_w)
                py2 = int(y2 / 999 * image_h)
                regions.append(
                    {
                        "type": label,
                        "bbox": [px1, py1, px2, py2],
                        "page": page_index,
                    }
                )
            except Exception:
                continue

    # Produce a markdown-friendly text: replace image refs, drop others
    cleaned = text
    for idx, m in enumerate(image_matches):
        cleaned = cleaned.replace(m, "\n")
    for m in other_matches:
        cleaned = (
            cleaned.replace(m, "")
            .replace("\\\n\\\n\\\n\\\n", "\\n\\n")
            .replace("\\\n\\\n\\\n", "\\n\\n")
            .replace("\\\\coloneqq", ":=")
            .replace("\\\\eqqcolon", "=:")
        )

    return {"text": cleaned}

def _build_mm_request(image: Image.Image, prompt: str) -> Dict[str, Any]:
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_features},
    }

def _generate_for_images(images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    if not images:
        return []
    # add prompt prefix
    prompt = PROMPT_PERFIX + prompt
    batch_inputs = [
        _build_mm_request(img, prompt) for img in images
    ]
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
    results: List[Dict[str, Any]] = []
    for idx, (out, img) in enumerate(zip(outputs_list, images)):
        content = out.outputs[0].text
        cleaned = _clean_text_and_collect_regions(content, img.width, img.height, idx)
        results.append(cleaned)
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Determine desired settings: prefer app.state (when launched via __main__),
    # otherwise fall back to environment (useful for import-string workers).
    model_path = getattr(app.state, "model_path", None) or os.environ.get("DS_MODEL_PATH")
    max_concurrency_env = os.environ.get("DS_MAX_CONCURRENCY")
    max_concurrency = getattr(app.state, "max_concurrency", None)
    if max_concurrency is None and max_concurrency_env is not None:
        try:
            max_concurrency = int(max_concurrency_env)
        except Exception:
            raise RuntimeError("Invalid DS_MAX_CONCURRENCY; must be int")

    # Initialize or validate LLM here; fail startup on error
    _ensure_llm(model_path=model_path, max_concurrency=max_concurrency)
    yield


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR API",
    description="Blazing fast OCR with DeepSeek-OCR model 🔥",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ocr")
async def ocr(
    file: UploadFile = File(...),
    prompt: str = Form(PROMPT),
    file_type: Optional[str] = Form(None),
    dpi: int = Form(144),
) -> JSONResponse:
    """
    Perform OCR on an uploaded PDF or image.

    - file: uploaded file (PDF or image)
    - prompt: optional prompt (defaults to config.PROMPT)
    - file_type: optional hint: "pdf" or "image"; auto-detects by content type/extension
    - dpi: PDF rasterization DPI (default 144)
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Start server with --model-path.")

    data = await file.read()
    content_type = file.content_type or ""
    name = file.filename or ""

    # Determine type
    t = (file_type or "").lower()
    is_pdf = False
    if t == "pdf":
        is_pdf = True
    elif t == "image":
        is_pdf = False
    else:
        if content_type == "application/pdf" or name.lower().endswith(".pdf"):
            is_pdf = True
        else:
            is_pdf = False

    try:
        if is_pdf:
            images = _pdf_bytes_to_images_high_quality(data, dpi=dpi)
            results = _generate_for_images(images, prompt)
            # Keep consistent page indexing starting at 0 in regions; also include count
            return JSONResponse({
                "type": "pdf",
                "pages": results,
                "num_pages": len(results),
            })
        else:
            image = _load_image_from_bytes(data)
            result = _generate_for_images([image], prompt)[0]
            return JSONResponse({
                "type": "image",
                **result,
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepSeek OCR REST API")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--model-path", required=False, default=os.environ.get("DS_MODEL_PATH"))
    parser.add_argument("--max-concurrency", type=int, required=False,
                        default=int(os.environ.get("DS_MAX_CONCURRENCY", 32)))
    parser.add_argument("--workers", type=int, required=False,
                        default=1,
                        help="Number of uvicorn worker processes (do not over-subscribe GPU)")
    args = parser.parse_args()

    # Propagate CLI config to env so worker processes (if any) can read them
    if args.model_path:
        os.environ["DS_MODEL_PATH"] = args.model_path
    os.environ["DS_MAX_CONCURRENCY"] = str(args.max_concurrency)

    # Initialize LLM per CLI args in this process (used when workers == 1)
    try:
        _ensure_llm(model_path=args.model_path, max_concurrency=args.max_concurrency)
    except Exception as e:
        raise SystemExit(f"Failed to initialize model: {e}")

    # Also store on app.state for the same-process server path
    app.state.model_path = args.model_path
    app.state.max_concurrency = args.max_concurrency

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.workers == 1:
        # Run with in-process app instance to preserve initialized model
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
    else:
        # Multiple workers require import string
        uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False, workers=args.workers)

