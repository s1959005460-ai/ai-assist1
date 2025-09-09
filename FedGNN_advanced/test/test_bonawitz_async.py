import asyncio
import numpy as np
import pytest
from FedGNN_advanced.privacy.bonawitz_async import AsyncServer, AsyncClient


@pytest.mark.asyncio
async def test_bonawitz_async_all_online_sum_equals_expected():
    # Setup
    server = AsyncServer(threshold_fraction=0.6)
    param_shapes = {"w": (2, 2), "b": (2,)}

    # Create & register clients
    clients = []
    for i in range(4):
        cid = f"client_{i}"
        c = AsyncClient(cid, server, param_shapes)
        await c.register()
        clients.append(c)

    # Phase: prepare and send shares
    for c in clients:
        await c.prepare_and_send_shares()

    # Each client collects shares addressed to them
    for c in clients:
        await c.collect_initial_shares(timeout=0.05)

    # All clients send masked updates
    updates = {}
    for idx, c in enumerate(clients):
        upd = {"w": np.ones((2, 2), dtype=np.float32) * (idx + 1), "b": np.ones((2,), dtype=np.float32) * (idx + 1)}
        updates[c.client_id] = upd
        await c.send_masked_update(upd)

    # Compute aggregate
    agg = await server.compute_aggregate()

    # Expected: sum of all updates (since everyone sent and masks cancel)
    expected = {}
    for p in updates[clients[0].client_id].keys():
        s = None
        for u in updates.values():
            if s is None:
                s = np.array(u[p], dtype=np.float64)
            else:
                s += np.array(u[p], dtype=np.float64)
        expected[p] = s.astype(np.float32)

    # compare
    for k in expected.keys():
        assert np.allclose(agg[k], expected[k], atol=1e-5)


@pytest.mark.asyncio
async def test_bonawitz_async_with_dropout_and_reconstruction():
    # Setup with one dropout
    server = AsyncServer(threshold_fraction=0.5)
    param_shapes = {"w": (2, 2), "b": (2,)}

    clients = []
    for i in range(5):
        cid = f"client_{i}"
        c = AsyncClient(cid, server, param_shapes)
        await c.register()
        clients.append(c)

    # Prepare & send shares
    for c in clients:
        await c.prepare_and_send_shares()

    # Collect incoming shares
    for c in clients:
        await c.collect_initial_shares(timeout=0.05)

    # Simulate dropout for client_3
    for c in clients:
        if c.client_id == "client_3":
            c.will_send_masked_update = False

    updates = {}
    for idx, c in enumerate(clients):
        upd = {"w": np.ones((2, 2), dtype=np.float32) * (idx + 1), "b": np.ones((2,), dtype=np.float32) * (idx + 1)}
        updates[c.client_id] = upd
        await c.send_masked_update(upd)

    # Server requests unmasking and clients provide their stored shares
    missing = server.missing_clients()
    assert "client_3" in missing

    # Explicitly collect unmask shares from clients and push them to server
    for c in clients:
        us = await c.provide_unmask_shares(missing)
        for item in us:
            await server._collect_unmask_share(item)

    # Now compute aggregate (should succeed if reconstruction possible)
    agg = await server.compute_aggregate()

    # Expected: sum of updates for clients who actually sent (i != 3)
    expected = None
    for cid, u in updates.items():
        if cid == "client_3":
            continue
        if expected is None:
            expected = {p: np.array(v, dtype=np.float64) for p, v in u.items()}
        else:
            for p, v in u.items():
                expected[p] += np.array(v, dtype=np.float64)
    for p in expected:
        assert np.allclose(agg[p], expected[p].astype(np.float32), atol=1e-5)
