"""
bonawitz_async.py

Asyncio-based Bonawitz-style secure aggregation protocol (single-process simulation).

Provides:
- Async Server and Client classes that talk via asyncio.Queue (inbox).
- Deterministic mask derivation via mask_manager + shamir share splitting/reconstruction.
- Clients store incoming SharePackage (including sender info) so they can later provide
  UnmaskShare for missing clients upon server request.
- Server orchestrates registration, share relay, masked update collection, unmask request,
  reconstruction and final aggregation.

Notes:
- This is a simulation-friendly, async-integration-ready implementation.
- In production you would replace queue.put/queue.get with network RPC (websockets / grpc).
"""

from typing import Dict, List, Tuple, Any, Optional
import asyncio
import numpy as np
from dataclasses import dataclass
from . import protocol_messages as msg
from . import shamir, mask_manager
from .. import dp_rng, constants, logger

SEED_BYTE_LEN = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)


# ---- helper dataclasses for internal bookkeeping ----
@dataclass
class ClientHandle:
    client_id: str
    inbox: asyncio.Queue  # Queue[msg.*]
    param_shapes: Dict[str, Tuple[int, ...]]


# ---- Server ----
class AsyncServer:
    def __init__(self, threshold_fraction: Optional[float] = None):
        self.clients: Dict[str, ClientHandle] = {}
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.masked_updates: Dict[str, Dict[str, Any]] = {}
        # store shares relayed (keeps history with sender info)
        self._relayed_shares: List[msg.SharePackage] = []
        self._unmask_shares: List[msg.UnmaskShare] = []
        self.threshold_fraction = (
            threshold_fraction
            if threshold_fraction is not None
            else constants.DEFAULTS.get("SHAMIR_THRESHOLD_FRACTION", 0.5)
        )
        # optional per-server lock for concurrency safety
        self._lock = asyncio.Lock()

    async def register_client(self, register_msg: msg.RegisterClient, inbox: asyncio.Queue):
        """
        Register a client; inbox is where server will place messages destined for the client.
        """
        cid = register_msg.client_id
        async with self._lock:
            self.clients[cid] = ClientHandle(client_id=cid, inbox=inbox, param_shapes=register_msg.param_shapes)
            # record param shapes; assume consistent across clients (overwritten but equal)
            self.param_shapes = register_msg.param_shapes
        logger.secure_log("info", f"Registered client {cid}", client_id=cid)

    def active_clients(self) -> List[str]:
        return list(self.clients.keys())

    def expected_threshold(self) -> int:
        n = len(self.clients)
        frac = self.threshold_fraction
        thr = int(np.ceil(n * float(frac)))
        return max(1, thr)

    # ---- share relay (async) ----
    async def relay_share(self, share_pkg: msg.SharePackage):
        """
        Place share package into recipient's inbox queue and record with sender info.
        """
        recipient = share_pkg.recipient
        async with self._lock:
            if recipient not in self.clients:
                logger.secure_log("warning", "Relay target unknown", recipient=recipient)
                return
            # Put package in recipient's inbox
            await self.clients[recipient].inbox.put(share_pkg)
            # keep a record for server-side diagnostics (sender included)
            self._relayed_shares.append(share_pkg)
        logger.secure_log("debug", "Relayed share", sender=share_pkg.sender, recipient=share_pkg.recipient)

    # ---- receive masked updates (async) ----
    async def receive_masked_update(self, masked_update: msg.MaskedUpdate):
        async with self._lock:
            self.masked_updates[masked_update.sender] = masked_update.masked_params
        logger.secure_log("info", "Server received masked update", sender=masked_update.sender)

    # ---- request unmasking: ask clients to provide their stored shares for missing clients ----
    async def request_unmasking(self, timeout: float = 5.0):
        """
        Ask each active client to provide UnmaskShare objects for missing clients.
        We gather responses concurrently (with timeout). Collected UnmaskShares are stored server-side.
        """
        missing = self.missing_clients()
        if not missing:
            logger.secure_log("info", "No missing clients; skipping unmask request")
            return

        logger.secure_log("info", "Requesting unmask shares", missing_clients=missing)
        tasks = []
        async with self._lock:
            for cid, handle in self.clients.items():
                # send an UnmaskRequest into client's inbox as a control message,
                # but in this in-process sim we'll directly call a coroutine that the client exposes.
                # We'll expect clients to register an asyncio-compatible handler (see Client.handle_unmask_request)
                # For decoupling, we call client handler via a convention: clients expose 'handle_unmask_request' coroutine.
                # To avoid tight coupling, server will rely on client object to implement it (server doesn't hold client object).
                # Instead, here we simulate by sending a special msg.UnmaskRequest to inbox and letting client loop respond.
                await handle.inbox.put(msg.UnmaskRequest(requester="server", missing_clients=missing))
            # Now, wait for unmask shares to be pushed back into server._unmask_shares by clients (they should call server._collect_unmask_share)
        # Wait a small period for clients to respond; in a real network you'd await responses per-client RPC.
        await asyncio.sleep(0.01)  # give clients a tick to process
        # Caller may later call reconstruct_missing_seeds; unmask_shares will be filled as clients respond.

    async def _collect_unmask_share(self, unmask_share: msg.UnmaskShare):
        """
        Internal API for clients to push UnmaskShare back to server.
        (Clients will call this; it's expected to be async.)
        """
        async with self._lock:
            self._unmask_shares.append(unmask_share)
        logger.secure_log("debug", "Server collected unmask share", sender=unmask_share.sender, missing_client=unmask_share.missing_client)

    def participants_who_sent(self) -> List[str]:
        return list(self.masked_updates.keys())

    def missing_clients(self) -> List[str]:
        return [c for c in self.clients.keys() if c not in self.masked_updates]

    async def reconstruct_missing_seeds(self) -> Dict[str, bytes]:
        """
        Attempt to reconstruct missing clients' master seeds using collected unmask shares.
        Returns dict missing_client -> seed_bytes for reconstructed ones.
        """
        reconstructed = {}
        # group shares by missing_client
        shares_by_target: Dict[str, List[Tuple[int, int, str]]] = {}
        async with self._lock:
            for us in self._unmask_shares:
                target = us.missing_client
                shares_by_target.setdefault(target, []).append((us.share[0], us.share[1], us.sender))

        for target, share_list in shares_by_target.items():
            thr = self.expected_threshold()
            if len(share_list) < thr:
                logger.secure_log("warning", "Not enough shares to reconstruct", target=target, have=len(share_list), need=thr)
                continue
            # choose first thr shares
            chosen = share_list[:thr]
            xs = [int(x) for (x, _, _) in [(c[0], c[1], c[2]) for c in chosen]]
            ys = [int(y) for (_, y, _) in [(c[0], c[1], c[2]) for c in chosen]]
            pairs = [(int(x), int(y)) for x, y in zip(xs, ys)]
            try:
                seed_bytes = shamir.reconstruct_secret_bytes(pairs, SEED_BYTE_LEN)
                reconstructed[target] = seed_bytes
                logger.secure_log("info", "Reconstructed missing seed", target=target)
            except Exception as e:
                logger.secure_log("warning", "Failed to reconstruct seed", target=target, err=str(e))
                continue
        return reconstructed

    async def compute_aggregate(self) -> Dict[str, Any]:
        """
        Compute the unmasked aggregate, attempting to reconstruct missing seeds if needed.
        """
        if not self.masked_updates:
            raise RuntimeError("No masked updates available")

        param_names = list(next(iter(self.masked_updates.values())).keys())
        agg = {p: None for p in param_names}

        # sum masked updates
        for sender, params in self.masked_updates.items():
            for p, arr in params.items():
                a = np.array(arr, dtype=np.float64)
                if agg[p] is None:
                    agg[p] = a.copy()
                else:
                    agg[p] += a

        missing = self.missing_clients()
        reconstructed_seeds = {}
        if missing:
            reconstructed_seeds = await self.reconstruct_missing_seeds()

        alive = [c for c in self.clients.keys() if c not in missing]
        # For each missing client m, if we reconstructed seed_m, remove its mask contributions
        for m in missing:
            seed_m = reconstructed_seeds.get(m)
            if seed_m is None:
                logger.secure_log("warning", "Cannot reconstruct missing client's seed; aggregation may be impossible", missing_client=m)
                continue
            for a in alive:
                pair_seed = mask_manager.derive_pairwise_seed(m, a, seed_m)
                masks = mask_manager.compute_mask_from_seed(pair_seed, self.param_shapes)
                for p, arr in masks.items():
                    if p in agg and agg[p] is not None:
                        agg[p] -= arr

        final = {p: agg[p].astype(np.float32) for p in param_names}
        return final


# ---- Client (async) ----
class AsyncClient:
    def __init__(self, client_id: str, server: AsyncServer, param_shapes: Dict[str, Tuple[int, ...]]):
        """
        client_id: unique id
        server: AsyncServer instance (in-process)
        param_shapes: parameter name -> shape tuples
        """
        self.client_id = str(client_id)
        self.server = server
        self.param_shapes = param_shapes
        self.inbox: asyncio.Queue = asyncio.Queue()
        # per-client master secret seed (bytes)
        rng = dp_rng.get_numpy_rng()
        self.master_seed: bytes = rng.bytes(constants.DEFAULTS.get("SEED_BYTE_LEN", 32))
        # mapping origin_sender -> list of shares (idx, share_int) that this client received during initial share distribution
        self.received_shares_of_others: Dict[str, List[Tuple[int, int]]] = {}
        # list for shares addressed to me (raw tuple list)
        self.received_shares_for_me: List[Tuple[int, int]] = []
        # whether client will actually send masked update (simulate dropout)
        self.will_send_masked_update = True
        # register with server asynchronously
        self._register_task = None

    async def register(self):
        await self.server.register_client(msg.RegisterClient(client_id=self.client_id, param_shapes=self.param_shapes), self.inbox)

    async def prepare_and_send_shares(self):
        """
        Split master_seed and send share packages to all participants via server.relay_share.
        """
        n = len(self.server.clients)
        t = self.server.expected_threshold()
        shares = shamir.split_secret_bytes(self.master_seed, n=n, t=t)
        client_ids = list(self.server.clients.keys())
        for (idx, share_val), recipient in zip(shares, client_ids):
            pkg = msg.SharePackage(sender=self.client_id, recipient=recipient, share=(idx, int(share_val)))
            await self.server.relay_share(pkg)
        logger.secure_log("info", "Client sent Shamir shares", sender=self.client_id)

    async def collect_initial_shares(self, timeout: float = 1.0):
        """
        Drain inbox for SharePackage messages and store mapping sender -> share.
        This will collect SharePackage for which this client is the recipient.
        """
        # Keep draining until a timeout occurs (no new messages)
        while True:
            try:
                pkg = await asyncio.wait_for(self.inbox.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            if isinstance(pkg, msg.SharePackage):
                # store mapping: origin sender -> share
                self.received_shares_of_others.setdefault(pkg.sender, []).append(pkg.share)
                self.received_shares_for_me.append(pkg.share)
            else:
                # put back non-share messages for other handlers (like UnmaskRequest)
                # We push it back into queue's front by re-enqueuing; simple approach: store in a temp buffer
                # For simplicity we handle UnmaskRequest/other msgs in process_inbox loop instead
                # Here push it onto a temporary queue that handle_incoming will later process
                await self.inbox.put(pkg)
                break
        logger.secure_log("info", "Client collected initial shares", client=self.client_id, count=len(self.received_shares_for_me))

    async def send_masked_update(self, local_update: Dict[str, np.ndarray]):
        """
        Compute masked update and send to server. This will apply pairwise masks:
        - For pair (i,j): convention used: if i < j (lexicographic), i adds +mask, j adds -mask.
        """
        if not self.will_send_masked_update:
            logger.secure_log("info", "Client skipping masked update (simulated dropout)", client=self.client_id)
            return

        client_ids = list(self.server.clients.keys())
        masked = {}
        for p, arr in local_update.items():
            masked[p] = np.array(arr, dtype=np.float64)

        for other in client_ids:
            if other == self.client_id:
                continue
            pair_seed = mask_manager.derive_pairwise_seed(self.client_id, other, self.master_seed)
            masks = mask_manager.compute_mask_from_seed(pair_seed, self.param_shapes)
            sign = 1.0 if self.client_id < other else -1.0
            for p, arr in masks.items():
                masked[p] += sign * arr

        mu = msg.MaskedUpdate(sender=self.client_id, masked_params={k: v.astype(np.float32) for k, v in masked.items()})
        await self.server.receive_masked_update(mu)
        logger.secure_log("info", "Client sent masked update", sender=self.client_id)

    async def handle_incoming_loop(self):
        """
        Optional background task: process control messages such as UnmaskRequest.
        This routine should be awaited or scheduled as a background task.
        """
        while True:
            pkg = await self.inbox.get()
            # termination sentinel support: if pkg is None, break
            if pkg is None:
                break
            if isinstance(pkg, msg.UnmaskRequest):
                # server asked for unmask shares for missing clients
                missing = pkg.missing_clients
                # prepare UnmaskShare objects for each missing client for which this client holds a share
                out_shares: List[msg.UnmaskShare] = []
                for m in missing:
                    # if this client received a share from origin == m, include it
                    # we expect received_shares_of_others to be keyed by origin sender
                    if m in self.received_shares_of_others:
                        # choose the first share we received from m (there should be exactly one)
                        for s in self.received_shares_of_others[m]:
                            us = msg.UnmaskShare(sender=self.client_id, missing_client=m, share=(int(s[0]), int(s[1])))
                            out_shares.append(us)
                # push unmask shares back to server via server internal API
                for us in out_shares:
                    await self.server._collect_unmask_share(us)
            else:
                # unknown control message; ignore or log
                logger.secure_log("debug", "Client received non-control message", client=self.client_id, msg_type=type(pkg).__name__)

    async def provide_unmask_shares(self, missing_clients: List[str]) -> List[msg.UnmaskShare]:
        """
        Backwards-compatible coroutine: return UnmaskShare list for missing clients.
        Implementation uses stored received_shares_of_others mapping.
        """
        out: List[msg.UnmaskShare] = []
        for m in missing_clients:
            if m in self.received_shares_of_others:
                for s in self.received_shares_of_others[m]:
                    out.append(msg.UnmaskShare(sender=self.client_id, missing_client=m, share=(int(s[0]), int(s[1]))))
        return out

    def stop_inbox_loop(self):
        """
        Enqueue sentinel None into inbox to stop handle_incoming_loop.
        """
        self.inbox.put_nowait(None)

