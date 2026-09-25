"""Command-line interface: `dfre`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .api.facade import DFRE
from .calibration.parameters import calibrate
from .calibration.thresholds import calibrate_thresholds, to_policy


def _cmd_score(args: argparse.Namespace) -> int:
    result = DFRE(lam=args.lam, gamma=args.gamma).score(
        args.missing, args.invalid, args.drift, args.uncertainty, n=args.n,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"R = {result.risk:.4f}  band={result.band}  action={result.action}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    signals = {}
    for pair in args.signal:
        name, _, value = pair.partition("=")
        signals[name] = float(value)
    model = DFRE(lam=args.lam, gamma=args.gamma)
    e = model.explain(signals, n=args.n)
    print(e.text)
    if args.json:
        print(json.dumps(e.to_dict(), indent=2))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import pandas as pd
    history = pd.read_csv(args.history)
    result = calibrate(history, label_col=args.label,
                       cv_folds=args.cv_folds, bootstrap=args.bootstrap)
    out = {"lam": result.lam, "gamma": result.gamma, "auc": result.auc}
    if result.auc_ci:
        out["auc_ci95"] = list(result.auc_ci)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_calibrate_thresholds(args: argparse.Namespace) -> int:
    import pandas as pd
    from .core.equation import dfre_array
    df = pd.read_csv(args.history)
    scores = dfre_array(
        df["M"].to_numpy(), df["V"].to_numpy(),
        df["D"].to_numpy(), df["U"].to_numpy(),
        n=args.n, lam=args.lam, gamma=args.gamma,
    )
    result = calibrate_thresholds(
        df[args.label].to_numpy(), scores,
        cost_fp=args.cost_fp, cost_fn=args.cost_fn,
    )
    policy = to_policy(result)
    print(json.dumps(policy.to_dict(), indent=2))
    return 0


def _cmd_trend(args: argparse.Namespace) -> int:
    from .monitoring.history import HistoryStore
    from .monitoring.trends import TrendDetector
    store = HistoryStore(args.history)
    report = TrendDetector(window=args.window).analyze(store.risks())
    print(json.dumps(report.__dict__, indent=2))
    if report.alarm:
        print("ALARM: upward risk trend detected", file=sys.stderr)
        return 2
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .monitoring.history import HistoryStore
    from .report.html import generate_html_report
    records = HistoryStore(args.history).load()
    html = generate_html_report(records, title=args.title)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Report written to {args.out} ({len(records)} batches)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .integrations.fastapi_ext import create_app
    import uvicorn
    model = DFRE.load(args.model) if args.model else DFRE()
    uvicorn.run(create_app(model), host=args.host, port=args.port)
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from ._version import __version__
    print(f"dfre {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dfre", description="DFRE CLI")
    sub = parser.add_subparsers(dest="command")

    p_score = sub.add_parser("score", help="Score a single batch")
    p_score.add_argument("--missing", type=float, required=True)
    p_score.add_argument("--invalid", type=float, required=True)
    p_score.add_argument("--drift", type=float, required=True)
    p_score.add_argument("--uncertainty", type=float, required=True)
    p_score.add_argument("--n", type=int, default=1000)
    p_score.add_argument("--lam", type=float, default=1.0)
    p_score.add_argument("--gamma", type=float, default=1.0)
    p_score.add_argument("--json", action="store_true")
    p_score.set_defaults(func=_cmd_score)

    p_exp = sub.add_parser("explain", help="Attribute risk to signals")
    p_exp.add_argument("--signal", action="append", required=True,
                       metavar="NAME=VALUE", help="e.g. --signal M=0.4 --signal V=0.1")
    p_exp.add_argument("--n", type=int, default=1000)
    p_exp.add_argument("--lam", type=float, default=1.0)
    p_exp.add_argument("--gamma", type=float, default=1.0)
    p_exp.add_argument("--json", action="store_true")
    p_exp.set_defaults(func=_cmd_explain)

    p_cal = sub.add_parser("calibrate", help="Calibrate lambda, gamma from history")
    p_cal.add_argument("--history", required=True)
    p_cal.add_argument("--label", default="failed")
    p_cal.add_argument("--cv-folds", type=int, default=1)
    p_cal.add_argument("--bootstrap", type=int, default=0)
    p_cal.set_defaults(func=_cmd_calibrate)

    p_thr = sub.add_parser("calibrate-thresholds", help="Calibrate band thresholds")
    p_thr.add_argument("--history", required=True)
    p_thr.add_argument("--label", default="failed")
    p_thr.add_argument("--n", type=int, default=1000)
    p_thr.add_argument("--lam", type=float, default=1.0)
    p_thr.add_argument("--gamma", type=float, default=1.0)
    p_thr.add_argument("--cost-fp", type=float, default=1.0)
    p_thr.add_argument("--cost-fn", type=float, default=10.0)
    p_thr.set_defaults(func=_cmd_calibrate_thresholds)

    p_trend = sub.add_parser("trend", help="Detect risk trend from history JSONL")
    p_trend.add_argument("--history", required=True, help="History JSONL file")
    p_trend.add_argument("--window", type=int, default=20)
    p_trend.set_defaults(func=_cmd_trend)

    p_rep = sub.add_parser("report", help="Render history JSONL to an HTML report")
    p_rep.add_argument("--history", required=True, help="History JSONL file")
    p_rep.add_argument("--out", default="dfre_report.html")
    p_rep.add_argument("--title", default="DFRE Monitoring Report")
    p_rep.set_defaults(func=_cmd_report)

    p_srv = sub.add_parser("serve", help="Serve scoring over HTTP (FastAPI)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--model", default=None, help="Saved DFRE JSON")
    p_srv.set_defaults(func=_cmd_serve)

    p_info = sub.add_parser("info", help="Show version")
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
