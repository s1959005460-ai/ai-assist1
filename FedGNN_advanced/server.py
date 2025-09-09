"""
FedGNN_advanced/server.py

FedServer: an async orchestration wrapper that integrates:
- bonawitz_async.AsyncServer (secure aggregation flow)
- Aggregator (robust weighted aggregation)
- compression instrumentation
- monitoring / logging
- evaluation (accuracy/precision/recall/f1) w/ results.json persist

Usage (in-process demo):
    from FedGNN_advanced.privacy.bonawitz_async import AsyncServer, AsyncClient
    from FedGNN_advanced.aggregator import Aggregator
    from FedGNN_advanced.server import FedServer
    # create model, server, client_objects, and pass client_update_fn
    fs = FedServer(global_model=model, bonawitz_server=async_server,
                   aggregator=Aggregator(device='cpu'),
                   client_objects=client_map, client_update_fn=my_update_fn, device='cpu')
    await fs.run_rounds(rounds=10, test_data=centralized_test_data)
"""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from . import logger, constants, dp_rng
from . import compression, monitoring
from .aggregator import Aggregator, client_delta_cosine_scores
from .privacy.bonawitz_async import AsyncServer  # the async protocol server we created earlier

# typing for client update function:
# can be sync def f(client_obj) -> Dict[param_name -> ndarray]
# or async def f(client_obj) -> Dict[...]
ClientUpdateFnType = Callable[[Any], Any]


class FedServer:
    def __init__(
        self,
        global_model: torch.nn.Module,
        bonawitz_server: AsyncServer,
        aggregator: Aggregator,
        client_objects: Dict[str, Any],
        client_update_fn: ClientUpdateFnType,
        device: str = "cpu",
        results_dir: str | Path = "runs",
        clients_per_round: Optional[int] = None,
    ):
        """
        Args:
            global_model: torch.nn.Module (uninitialized device ok)
            bonawitz_server: AsyncServer instance (in-process orchestrator)
            aggregator: Aggregator instance
            client_objects: mapping client_id -> client_object (e.g. AsyncClient instances)
            client_update_fn: callable(client_obj) -> local_update_dict (param_name -> ndarray or torch tensor)
                Can be async or sync. For real networked clients, replace this with RPC wrappers that trigger remote training.
            device: device for model evaluation / parameter loading
            results_dir: directory to save results.json
            clients_per_round: if None, use all clients; else sample this many per round
        """
        self.global_model = global_model
        self.bonawitz_server = bonawitz_server
        self.aggregator = aggregator
        self.client_objects = client_objects
        self.client_update_fn = client_update_fn
        self.device = torch.device(device)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "results.json"
        if not self.results_file.exists():
            with open(self.results_file, "w") as f:
                json.dump([], f)
        self.clients_per_round = clients_per_round

    # -------------------------
    # Utility IO
    # -------------------------
    def _append_results(self, info: Dict[str, Any]):
        with open(self.results_file, "r+") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
            data.append(info)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    # -------------------------
    # Client selection
    # -------------------------
    def _select_clients(self) -> List[str]:
        all_clients = list(self.bonawitz_server.clients.keys())
        if not all_clients:
            return []
        if self.clients_per_round is None or self.clients_per_round >= len(all_clients):
            return all_clients
        # sample deterministically from dp_rng
        rng = np.random.default_rng(dp_rng.current_seed() or 0)
        chosen = rng.choice(all_clients, size=self.clients_per_round, replace=False).tolist()
        return chosen

    # -------------------------
    # Core round orchestration
    # -------------------------
    async def run_rounds(
        self,
        rounds: int,
        test_data: Any,
        seed: Optional[int] = None,
        bootstrap_shares: bool = True,
    ):
        """
        Run federated rounds.

        Args:
            rounds: number of rounds to run
            test_data: dataset wrapper for evaluate_global_model (must have x, edge_index, y, test_mask)
            seed: optional RNG seed
            bootstrap_shares: whether to run initial Shamir share distribution at the beginning (True for first-run)
        """
        if seed is not None:
            dp_rng.set_seed(seed)
            logger.secure_log("info", "Seed set for experiment", seed=seed)

        # If we run shares at start, call prepare_and_send_shares for each client and let them collect shares.
        if bootstrap_shares:
            logger.secure_log("info", "Bootstrapping Shamir shares among clients")
            # clients are expected to be objects with prepare_and_send_shares and collect_initial_shares coroutines
            tasks = []
            for cid, client in self.client_objects.items():
                if hasattr(client, "prepare_and_send_shares"):
                    tasks.append(asyncio.create_task(client.prepare_and_send_shares()))
                else:
                    logger.secure_log("warning", "Client missing prepare_and_send_shares", client=cid)
            # wait
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

            # Now let clients collect their incoming shares
            tasks = []
            for cid, client in self.client_objects.items():
                if hasattr(client, "collect_initial_shares"):
                    tasks.append(asyncio.create_task(client.collect_initial_shares(timeout=0.1)))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

        for r in range(1, rounds + 1):
            round_start = time.time()
            selected = self._select_clients()
            logger.secure_log("info", "Selected clients for round", round=r, selected=selected)

            # Ask each selected client to (a) compute local update and (b) send masked update
            # We'll call client_update_fn to obtain plaintext update (for debugging/monitoring/anomaly stats).
            # client_update_fn may be async or sync.
            client_plain_updates: Dict[str, Dict[str, Any]] = {}
            client_tasks = []
            for cid in selected:
                client = self.client_objects.get(cid)
                if client is None:
                    logger.secure_log("warning", "Selected client not found in client_objects", client=cid)
                    continue

                # call user-provided client_update_fn(client) to produce local_update
                if asyncio.iscoroutinefunction(self.client_update_fn):
                    t = asyncio.create_task(self.client_update_fn(client))
                else:
                    # run sync function in threadpool to avoid blocking asyncio loop
                    loop = asyncio.get_running_loop()
                    t = loop.run_in_executor(None, self.client_update_fn, client)
                client_tasks.append((cid, client, t))

            # gather updates
            for cid, client, task in client_tasks:
                try:
                    local_update = await task  # dict param->ndarray or torch tensor
                except Exception as e:
                    logger.secure_log("warning", "client_update_fn failed", client=cid, err=str(e))
                    continue
                # keep plaintext for monitoring/aggregation debugging
                client_plain_updates[cid] = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.array(v)) for k, v in local_update.items()}

                # Now instruct client to send masked update (in-process API)
                # If client is remote, replace this with RPC call to submit masked update
                if hasattr(client, "send_masked_update"):
                    try:
                        await client.send_masked_update(local_update)
                    except Exception as e:
                        logger.secure_log("warning", "client.send_masked_update failed", client=cid, err=str(e))
                else:
                    logger.secure_log("warning", "Client has no send_masked_update API; skipping masked upload", client=cid)

            # At this point, bonawitz_async.AsyncServer should have stored masked updates in bonawitz_server.masked_updates.
            # Wait briefly to let messages propagate (in networked env you'll await RPC responses instead)
            await asyncio.sleep(0.01)

            # If some selected clients didn't send (dropout), request unmasking and collect unmask shares
            missing = self.bonawitz_server.missing_clients()
            if missing:
                logger.secure_log("info", "Detected missing clients; requesting unmasking", missing_clients=missing)
                await self.bonawitz_server.request_unmasking(timeout=1.0)
                # In our in-process sim, clients will respond via their inbox; to ensure server has shares, call provide_unmask_shares on all alive clients
                for cid, client in self.client_objects.items():
                    if hasattr(client, "provide_unmask_shares"):
                        try:
                            shares = await client.provide_unmask_shares(missing)
                            for us in shares:
                                await self.bonawitz_server._collect_unmask_share(us)
                        except Exception as e:
                            logger.secure_log("warning", "provide_unmask_shares failed", client=cid, err=str(e))

            # Now attempt reconstruction & aggregation
            try:
                unmasked_agg = await self.bonawitz_server.compute_aggregate()
            except Exception as e:
                logger.secure_log("error", "Aggregation failed", err=str(e))
                # fallback: skip round
                round_info = {"round": r, "error": str(e), "timestamp": time.time()}
                self._append_results(round_info)
                continue

            # unmasked_agg is dict param_name -> numpy array (float32)
            # Optionally compute compression metadata for each client's update vs compressed size
            compression_meta = {}
            # If we have plaintext client updates, compute compression stats relative to each client's plain update
            for cid, upd in client_plain_updates.items():
                comp_info = {}
                for pname, arr in upd.items():
                    try:
                        payload, meta = compression.serialize_sparse(arr)
                        comp_info[pname] = meta
                    except Exception as e:
                        comp_info[pname] = {"error": str(e)}
                compression_meta[cid] = comp_info

            # anomaly detection: if client_plain_updates present, compute cosine similarity scores
            anomaly_scores = {}
            if client_plain_updates:
                # convert client_plain_updates into mapping client_id -> delta_state_dict
                try:
                    anomaly_scores = client_delta_cosine_scores({}, client_plain_updates)
                except Exception as e:
                    logger.secure_log("warning", "anomaly detection failed", err=str(e))

            # Apply aggregated parameters to global_model
            try:
                # create a state_dict-like mapping with torch tensors
                new_state = {}
                for k, v in unmasked_agg.items():
                    new_state[k] = torch.tensor(v, device=self.device)
                # load into model non-strictly (to allow partial names)
                self.global_model.load_state_dict(new_state, strict=False)
            except Exception as e:
                logger.secure_log("error", "Failed to load aggregated params into global_model", err=str(e))

            # Evaluate
            try:
                metrics = self.evaluate_global_model(test_data, device=self.device)
            except Exception as e:
                logger.secure_log("warning", "Evaluation failed", err=str(e))
                metrics = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

            round_time = time.time() - round_start
            round_info = {
                "round": r,
                "timestamp": time.time(),
                "round_time_seconds": round_time,
                "num_selected": len(selected),
                "num_missing": len(missing),
                "metrics": metrics,
                "compression_meta": compression_meta,
                "anomaly_scores": anomaly_scores,
            }

            # monitoring: log to W&B / fallback logger
            monitoring.log_metrics(r, {"test_accuracy": metrics.get("accuracy", None), "test_f1": metrics.get("f1", None), "round_time": round_time})

            # persist round_info
            self._append_results(round_info)
            logger.secure_log("info", "Round complete", round=r, summary={"accuracy": metrics.get("accuracy"), "f1": metrics.get("f1")})

        logger.secure_log("info", "All rounds completed", rounds=rounds)

    # -------------------------
    # Evaluation helper (identical to earlier evaluate_global_model)
    # -------------------------
    def evaluate_global_model(self, test_data, device: Optional[str | torch.device] = None) -> Dict[str, float]:
        """
        Evaluate self.global_model on test_data and return metrics dict.
        test_data must expose: x, edge_index, y, test_mask (torch tensors).
        Returns: dict with accuracy, precision, recall, f1
        """
        if device is None:
            device = self.device
        device = torch.device(device)
        self.global_model.to(device)
        self.global_model.eval()

        x = test_data.x.to(device)
        edge_index = test_data.edge_index.to(device)
        y = test_data.y.to(device)
        mask = test_data.test_mask.to(device)

        with torch.no_grad():
            out = self.global_model(x, edge_index)  # (N, C)
            logits = out[mask]
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y[mask].cpu().numpy()

        # handle empty test masks gracefully
        if labels.size == 0:
            metrics = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            logger.secure_log("warning", "Empty test mask in evaluate_global_model")
            return metrics

        # sklearn metrics with safe zero division handling
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            accuracy = float((preds == labels).mean())
            precision = float(precision_score(labels, preds, average="macro", zero_division=0))
            recall = float(recall_score(labels, preds, average="macro", zero_division=0))
            f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        except Exception:
            # fallback simple accuracy
            accuracy = float((preds == labels).mean())
            precision = recall = f1 = float("nan")

        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        logger.secure_log("info", "Evaluation metrics", **metrics)
        return metrics

