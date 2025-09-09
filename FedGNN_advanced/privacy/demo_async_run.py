"""
demo_async_run.py

Simple demo to run the async Bonawitz flow in-process.

Usage:
    python -m FedGNN_advanced.privacy.demo_async_run
or
    python FedGNN_advanced/privacy/demo_async_run.py
(ensure FedGNN_advanced package is importable)
"""

import asyncio
import numpy as np
from .bonawitz_async import AsyncServer, AsyncClient
from .. import dp_rng, constants, logger

async def run_demo():
    dp_rng.set_seed(12345)
    server = AsyncServer(threshold_fraction=0.6)
    param_shapes = {"w": (2, 2), "b": (2,)}

    # Create clients
    clients = []
    for i in range(5):
        cid = f"client_{i}"
        c = AsyncClient(cid, server, param_shapes)
        await c.register()
        clients.append(c)

    # Phase: each client prepares/shares shamir shares
    for c in clients:
        await c.prepare_and_send_shares()

    # Phase: each client collects incoming shares
    for c in clients:
        # drain incoming share messages for a short time
        await c.collect_initial_shares(timeout=0.1)

    # Start inbox loops (to handle UnmaskRequest messages)
    tasks = [asyncio.create_task(c.handle_incoming_loop()) for c in clients]

    # Phase: clients send masked updates (simulate client_3 dropout)
    for c in clients:
        if c.client_id == "client_3":
            c.will_send_masked_update = False  # simulate dropout
        upd = {"w": np.ones((2,2), dtype=np.float32) * (int(c.client_id.split("_")[-1]) + 1), "b": np.ones((2,), dtype=np.float32)}
        await c.send_masked_update(upd)

    # Server sees missing clients and requests unmask shares
    await server.request_unmasking()
    # Instead of relying on background loops to push unmask shares, explicitly call provide_unmask_shares on each client
    missing = server.missing_clients()
    for c in clients:
        us = await c.provide_unmask_shares(missing)
        for item in us:
            await server._collect_unmask_share(item)

    # Compute aggregate
    agg = await server.compute_aggregate()
    print("Aggregated params:", {k: v.tolist() for k, v in agg.items()})

    # stop inbox loops
    for c in clients:
        c.stop_inbox_loop()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(run_demo())
