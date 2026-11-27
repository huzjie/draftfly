"""命令行入口：doctor / generate / bench / eval / serve / config / version。"""
from __future__ import annotations

import argparse
import json
import random
import sys

from . import __version__
from .bench import run_bench
from .config import Config
from .engine.speculator import Speculator
from .eval import run_eval
from .models.registry import build_draft, build_target
from .tokenizer import CharTokenizer
from .util.logger import Logger


DEFAULT_PROMPT = "Speculative decoding makes large language models faster by"


def _load_cfg(path):
    cfg = Config.load(path)
    errs = cfg.validate()
    if errs:
        raise SystemExit("配置错误:\n- " + "\n- ".join(errs))
    return cfg


def _build(cfg):
    tok = CharTokenizer()
    draft = build_draft(cfg.draft, tok)
    target = build_target(cfg.target, tok)
    return tok, draft, target


def cmd_doctor(cfg, log):
    log.info("draftfly doctor 自检开始")
    tok, draft, target = _build(cfg)
    spec = Speculator(
        draft, target, tok,
        k=cfg.engine.k, mode="greedy", temperature=1.0, seed=cfg.engine.seed,
    )
    res = spec.generate(DEFAULT_PROMPT, max_new_tokens=32)
    assert res.text, "生成结果为空"
    ev = run_eval(tok, DEFAULT_PROMPT, max_new_tokens=128, seed=cfg.engine.seed)
    checks = {
        "tokenizer.vocab_size": tok.vocab_size,
        "draft.order": cfg.draft.order,
        "target.order": cfg.target.order,
        "generate.ok": bool(res.text),
        "eval.monotonic": ev["monotonic"],
        "sample": res.text[:40] + "...",
    }
    for k, v in checks.items():
        log.info(f"  - {k}: {v}")
    if not ev["monotonic"]:
        log.error("接受率未随 skill 单调，评测失败")
        raise SystemExit(1)
    log.info("doctor 自检通过 ✅")
    return checks


def cmd_generate(cfg, args, log):
    tok, draft, target = _build(cfg)
    spec = Speculator(
        draft, target, tok,
        k=args.k or cfg.engine.k, mode=args.mode or cfg.engine.mode,
        temperature=cfg.engine.temperature, top_k=cfg.engine.top_k,
        top_p=cfg.engine.top_p, seed=cfg.engine.seed,
    )
    res = spec.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(res.text)
    log.info(f"acceptance_rate={res.acceptance_rate:.4f}  tokens={len(res.tokens)}")


def cmd_bench(cfg, args, log):
    tok, draft, target = _build(cfg)
    out = run_bench(
        draft, target, tok, args.prompt, max_new_tokens=args.max_tokens,
        k=args.k or cfg.engine.k, mode=cfg.engine.mode, temperature=cfg.engine.temperature,
        seed=cfg.engine.seed, real_clock=cfg.engine.real_clock,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    log.info(f"speedup={out['speedup']}x  acceptance={out['acceptance_rate']}")


def cmd_eval(cfg, args, log):
    tok, _, _ = _build(cfg)
    out = run_eval(tok, args.prompt, max_new_tokens=args.max_tokens, seed=cfg.engine.seed)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    log.info(f"monotonic={out['monotonic']}")


def cmd_serve(cfg, log):
    from .api.server import serve
    serve(cfg)


def cmd_config(cfg, args):
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="draftfly", description="投机解码加速引擎")
    ap.add_argument("--config", default=None, help="配置文件路径（YAML/JSON）")
    ap.add_argument("--version", action="store_true", help="打印版本")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="环境自检")
    p = sub.add_parser("generate", help="投机解码生成文本")
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--mode", choices=["sampling", "greedy"], default=None)

    p = sub.add_parser("bench", help="投机 vs 自回归吞吐对比")
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--k", type=int, default=None)

    p = sub.add_parser("eval", help="接受率单调性评测")
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=64)

    sub.add_parser("serve", help="起 OpenAI 兼容 HTTP 服务")
    sub.add_parser("config", help="打印解析后的配置")

    args = ap.parse_args(argv)
    if args.version:
        print(__version__)
        return 0

    log = Logger()
    if args.command is None:
        ap.print_help()
        return 0

    cfg = _load_cfg(args.config)
    if args.command == "doctor":
        cmd_doctor(cfg, log)
    elif args.command == "generate":
        cmd_generate(cfg, args, log)
    elif args.command == "bench":
        cmd_bench(cfg, args, log)
    elif args.command == "eval":
        cmd_eval(cfg, args, log)
    elif args.command == "serve":
        cmd_serve(cfg, log)
    elif args.command == "config":
        cmd_config(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
