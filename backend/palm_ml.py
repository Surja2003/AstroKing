from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PalmEmbeddingResult:
    status: str
    embedding: Optional[list[float]] = None
    dim: Optional[int] = None
    error: Optional[str] = None


_interpreter_lock = threading.Lock()
_interpreter = None
_input_details = None
_output_details = None


def _default_model_path() -> str:
    # Default to the repo-local exported embedding model.
    base_dir = os.path.dirname(__file__)
    candidate = os.path.abspath(os.path.join(base_dir, "..", "ml", "models", "hand_embedding_float16.tflite"))
    return candidate


def _get_model_path() -> str:
    return os.environ.get("PALM_EMBEDDING_MODEL", "").strip() or _default_model_path()


def _lazy_import_pil():
    from PIL import Image  # type: ignore

    return Image


def _lazy_import_numpy():
    import numpy as np  # type: ignore

    return np


def _lazy_load_interpreter():
    global _interpreter, _input_details, _output_details

    if _interpreter is not None:
        return _interpreter

    with _interpreter_lock:
        if _interpreter is not None:
            return _interpreter

        model_path = _get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Embedding model not found at '{model_path}'. "
                "Set PALM_EMBEDDING_MODEL to a .tflite path."
            )

        interpreter = None
        last_err: Optional[Exception] = None

        # Prefer tflite-runtime when installed.
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            interpreter = Interpreter(model_path=model_path)
        except Exception as e:
            last_err = e

        # Fall back to TensorFlow's built-in TFLite interpreter.
        if interpreter is None:
            try:
                import tensorflow as tf  # type: ignore

                interpreter = tf.lite.Interpreter(model_path=model_path)
                last_err = None
            except Exception as e:
                last_err = e

        if interpreter is None:
            raise RuntimeError(
                "Could not initialize a TFLite interpreter. "
                "Install 'tflite-runtime' (recommended) or 'tensorflow'."
            ) from last_err

        interpreter.allocate_tensors()
        _input_details = interpreter.get_input_details()
        _output_details = interpreter.get_output_details()

        _interpreter = interpreter
        return _interpreter


def _preprocess_image_bytes(image_bytes: bytes, img_size: int):
    np = _lazy_import_numpy()
    Image = _lazy_import_pil()

    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def get_embedding_dim(*, img_size: int = 224) -> Optional[int]:
    try:
        interpreter = _lazy_load_interpreter()
        out = _output_details[0]  # type: ignore[index]
        shape = out.get("shape")
        if shape is None:
            return None
        # shape is usually [1, D]
        if len(shape) == 2:
            return int(shape[1])
        return None
    except Exception:
        return None


def embed_image_bytes(image_bytes: bytes, *, img_size: int = 224) -> PalmEmbeddingResult:
    """Return an embedding vector using the exported TFLite embedding model.

    Designed to be safe to call from the API: returns a status + error instead
    of raising.
    """

    try:
        np = _lazy_import_numpy()

        interpreter = _lazy_load_interpreter()
        if not _input_details or not _output_details:
            return PalmEmbeddingResult(status="error", error="Interpreter not initialized")

        x = _preprocess_image_bytes(image_bytes, img_size)

        input_info = _input_details[0]
        input_index = int(input_info["index"])
        expected_dtype = input_info.get("dtype")

        # Some float16 exports expect float32 input; use whatever the interpreter asks for.
        if expected_dtype is not None:
            x = x.astype(expected_dtype)

        with _interpreter_lock:
            interpreter.set_tensor(input_index, x)
            interpreter.invoke()
            out_info = _output_details[0]
            out_index = int(out_info["index"])
            emb = interpreter.get_tensor(out_index)

        emb = np.asarray(emb)
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        emb = emb.astype(np.float32)

        return PalmEmbeddingResult(
            status="ok",
            embedding=emb.tolist(),
            dim=int(emb.shape[0]) if emb.ndim == 1 else None,
        )
    except FileNotFoundError as e:
        return PalmEmbeddingResult(status="unavailable", error=str(e))
    except RuntimeError as e:
        # Most commonly: no TFLite interpreter provider installed.
        return PalmEmbeddingResult(status="unavailable", error=str(e))
    except Exception as e:
        return PalmEmbeddingResult(status="error", error=str(e))
