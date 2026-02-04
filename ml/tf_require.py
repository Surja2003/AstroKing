import sys


def require_tensorflow():
    """Import TensorFlow with a helpful error message.

    This repo keeps ML deps isolated under `ml/` and recommends Python 3.11 on Windows.
    """

    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "TensorFlow is not available in this Python environment.\n\n"
            f"- Current Python: {py} ({sys.executable})\n"
            "- On Windows, TensorFlow commonly requires Python 3.11 for this project.\n\n"
            "Fix (recommended):\n"
            "  cd ml\n"
            "  py -3.11 -m venv .venv\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install --upgrade pip\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n\n"
            "Then run your command using that interpreter, e.g.:\n"
            "  .\\.venv\\Scripts\\python.exe infer_one.py --model <path> --image <path>\n\n"
            f"Original error: {e}"
        )
