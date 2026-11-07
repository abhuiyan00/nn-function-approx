"""Local web interface for the funcprox engine (optional: ``pip install -e ".[ui]"``).

Kept out of ``funcprox`` so the engine stays NumPy-only. Start with ``python -m funcprox ui``.
"""
from __future__ import annotations

import threading
import webbrowser

import funcprox  # noqa: F401  (first: it sets the BLAS thread limits before NumPy loads)


def serve(port: int = 8765, open_browser: bool = True) -> int:
    import uvicorn

    from .app import STATIC, create_app

    url = f"http://127.0.0.1:{port}"
    print(f"funcprox interface: {url}  (Ctrl+C to stop)")
    if not (STATIC / "index.html").exists():
        print('Frontend not built yet: run "npm --prefix ui ci" and "npm --prefix ui run build".')
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    # loopback only, on purpose: the API trains models and starts processes and has no login
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=1)
    return 0
