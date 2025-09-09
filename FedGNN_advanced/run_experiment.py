"""
run_experiment.py

Safe entrypoint: uses asyncio.run and centralized constants and dp_rng.
"""

import argparse
import asyncio
import importlib
import sys
from . import constants, dp_rng, logger


def locate_server_class():
    """
    Locate common server class names. Raise informative ImportError if not found.
    """
    possible = [
        ("server", "AsyncFedServer"),
        ("server", "FedServer"),
        ("server", "Server"),
        ("server.server", "AsyncFedServer"),
    ]
    for module_name, cls_name in possible:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    raise ImportError(
        "Server class not found. Ensure your server module exposes AsyncFedServer/FedServer/Server."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run FedGNN experiment")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logdir", type=str, default="runs")
    return p.parse_args()


async def main_async(rounds: int, seed: int | None, device: str, logdir: str):
    if seed is None:
        seed = constants.DEFAULTS.get("DEFAULT_SEED", 0)
    dp_rng.set_seed(seed)
    logger.secure_log("info", "Seed set", seed=seed)

    ServerClass = locate_server_class()
    # Try common constructor patterns
    try:
        server = ServerClass(config=None, device=device, logdir=logdir)
    except TypeError:
        try:
            server = ServerClass(None, device, logdir)
        except Exception:
            server = ServerClass()
            for attr, val in [("device", device), ("logdir", logdir), ("config", None)]:
                setattr(server, attr, val)

    # find run coroutine
    run_coro = None
    if hasattr(server, "run_rounds"):
        run_coro = server.run_rounds(rounds)
    elif hasattr(server, "run_round"):
        run_coro = server.run_round(rounds=rounds)
    elif hasattr(server, "run"):
        try:
            run_coro = server.run(rounds=rounds)
        except TypeError:
            run_coro = server.run()
    else:
        raise RuntimeError("Server instance does not expose run/run_round/run_rounds coroutine")

    await run_coro


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args.rounds, args.seed, args.device, args.logdir))
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
