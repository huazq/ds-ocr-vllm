import io
import os
import re
import ast
from typing import List, Dict, Any, Optional

import argparse
import uvicorn

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
DEFAULT_PROMPT = 'Convert the document to markdown.'
CROP_MODE = True
DEFAULT_MODEL_PATH="/mnt/models"
DEFAULT_MODEL_NAME="DeepSeek-OCR"

# Register custom model class with vLLM
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def _init_llm(model_name: str,
              model_path: str,
              max_concurrency: int,
              tensor_parallel_size: int) -> LLM:
    # model_path is parent dir of model,
    # we need pass model_path+model_name to vllm
    model = model_path + "/" + model_name
    llm_local = LLM(
        served_model_name=model_name,
        model=model,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=64,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_concurrency,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )
    return llm_local


# Global singleton LLM and its current settings
llm: Optional[LLM] = None
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

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR API",
    description="Blazing fast OCR with DeepSeek-OCR model 🔥",
    version="2.0.0",
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
    prompt: str = Form(DEFAULT_PROMPT),
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

    parser = argparse.ArgumentParser(description="DeepSeek OCR REST API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", required=False, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", required=False, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-concurrency", type=int, required=False,
                        default=32)
    parser.add_argument("--tensor-parallel-size", type=int, required=False,
                        default=1)

    args = parser.parse_args()

    # Initialize LLM per CLI args in this process (used when workers == 1)
    try:
        llm = _init_llm(args.model_name,
                        args.model_path,
                        args.max_concurrency,
                        args.tensor_parallel_size)
    except Exception as e:
        raise SystemExit(f"Failed to initialize model: {e}")

    # Run with in-process app instance to preserve initialized model
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
