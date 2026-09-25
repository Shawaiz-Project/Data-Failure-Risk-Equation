"""FastAPI microservice exposing DFRE scoring over HTTP.

Run directly:

    python -m dfre.integrations.fastapi_ext --port 8000

or via the CLI:  ``dfre serve --port 8000``
"""

from __future__ import annotations

from typing import Dict, Optional

from ..api.facade import DFRE


def create_app(model: Optional[DFRE] = None):
    """Build a FastAPI app with POST /score and GET /health.

    Requires ``pip install dfre[serve]``.
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as e:  # pragma: no cover
        raise ImportError("fastapi required; pip install dfre[serve]") from e

    model = model or DFRE()

    class ScoreRequest(BaseModel):
        signals: Dict[str, float] = Field(
            ..., description="Named normalized signals in [0, 1], e.g. M, V, D, U")
        n: int = Field(1000, gt=0)
        metadata: Optional[Dict[str, str]] = None

    app = FastAPI(title="DFRE scoring service", version="0.2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "lam": model.lam, "gamma": model.gamma}

    @app.post("/score")
    def score_endpoint(req: ScoreRequest):
        result = model.score_generalized(req.signals, n=req.n,
                                         metadata=dict(req.metadata or {}))
        return result.to_dict()

    return app


def main() -> None:  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Serve DFRE over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None,
                        help="Path to a saved DFRE JSON (lam/gamma/policy)")
    args = parser.parse_args()

    import uvicorn
    model = DFRE.load(args.model) if args.model else DFRE()
    uvicorn.run(create_app(model), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
