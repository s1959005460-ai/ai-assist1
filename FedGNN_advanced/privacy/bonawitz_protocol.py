"""
bonawitz_protocol.py

End-to-end Bonawitz-style secure aggregation flow (single-process, in-memory simulation).

Classes:
- Server: orchestrates the protocol, holds registered clients, relays shares, aggregates masked updates.
- Client: simulates a client that:
    1) registers with server (provides param shapes)
    2) creates a per-client master_seed (random bytes)
    3) splits its master_seed via Shamir and sends shares to other participants (through server)
    4) derives pairwise seeds with other clients, computes pairwise masks and masks its local update
    5) sends masked update to server
    6) later, participates in unmasking by providing Shamir shares for missing clients when requested

Notes:
- This is a functional simulation intended for experimentation and testing.
- For production, network transport, authenticated channels and secure key storage must be used.
"""

from typing import Dict, List, Tuple, Any
from . import protocol_messages as msg
from . import shamir, mask_manager
from .. import dp_rng, constants, logger
import numpy as np
import copy

# convenience alias
SEED_BYTE_LEN = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)


class Server:
    def __init__(self, threshold_fraction: float | None = None):
        self.clients: Dict[str, "Client"] = {}
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.masked_updates: Dict[str, Dict[str, Any]] = {}
        self.received_shares: List[msg.SharePackage] = []  # store shares temporarily
        self.unmask_shares: List[msg.UnmaskShare] = []
        self.threshold_fraction = (
            threshold_fraction
            if threshold_fraction is not None
            else constants.DEFAULTS.get("SHAMIR_THRESHOLD_FRACTION", 0.5)
        )

    def register_client(self, register_msg: msg.RegisterClient, client_obj: "Client"):
        cid = register_msg.client_id
        self.clients[cid] = client_obj
        # record param_shapes assuming all clients share same model param names/shapes
        self.param_shapes = register_msg.param_shapes
        logger.secure_log("info", f"Registered client {cid}", client_id=cid)

    def relay_share(self, share_pkg: msg.SharePackage):
        """
        In a real deployment this would send via private channels to the recipient.
        Here we directly append to the recipient's incoming queue.
        """
        self.received_shares.append(share_pkg)
        logger.secure_log("info", "Relayed share package", sender=share_pkg.sender, recipient=share_pkg.recipient)

    def collect_shares_for_recipient(self, recipient: str) -> List[Tuple[int, int]]:
        """
        Return all shares addressed to recipient and remove them from relay buffer.
        """
        out = []
        remain = []
        for p in self.received_shares:
            if p.recipient == recipient:
                out.append(p.share)
            else:
                remain.append(p)
        self.received_shares = remain
        return out

    def receive_masked_update(self, masked_update: msg.MaskedUpdate):
        self.masked_updates[masked_update.sender] = masked_update.masked_params
        logger.secure_log("info", "Received masked update", sender=masked_update.sender)

    def active_clients(self) -> List[str]:
        return list(self.clients.keys())

    def expected_threshold(self) -> int:
        n = len(self.clients)
        frac = self.threshold_fraction
        # require at least ceil(n * frac)
        thr = int(np.ceil(n * float(frac)))
        return max(1, thr)

    def participants_who_sent(self) -> List[str]:
        return list(self.masked_updates.keys())

    def missing_clients(self) -> List[str]:
        # clients that failed to send masked updates
        return [c for c in self.clients.keys() if c not in self.masked_updates]

    def request_unmasking(self):
        """
        Request unmasking shares for any missing clients (dropped clients).
        The server asks all *remaining* clients for their Shamir shares for each missing client.
        """
        missing = self.missing_clients()
        if not missing:
            logger.secure_log("info", "No missing clients; no unmasking required")
            return
        logger.secure_log("info", "Requesting unmask shares", missing_clients=missing)
        for client in self.clients.values():
            # ask each alive client to send their unmask shares for missing clients
            shares = client.provide_unmask_shares(missing)
            # collector will store them for reconstruction
            for sh in shares:
                self.unmask_shares.append(sh)

    def reconstruct_missing_seeds(self) -> Dict[str, bytes]:
        """
        Attempt to reconstruct missing clients' master seeds using collected unmask shares.
        Returns dict missing_client -> seed_bytes for reconstructed ones.
        """
        reconstructed = {}
        # group shares by missing_client
        shares_by_target: Dict[str, List[Tuple[int, int, str]]] = {}
        for us in self.unmask_shares:
            target = us.missing_client
            shares_by_target.setdefault(target, []).append((us.share[0], us.share[1], us.sender))

        for target, share_list in shares_by_target.items():
            # need at least threshold shares from distinct senders to reconstruct
            n = len(self.clients)
            thr = self.expected_threshold()
            if len(share_list) < thr:
                logger.secure_log("warning", "Not enough shares to reconstruct", target=target, have=len(share_list), need=thr)
                continue
            # use first 'thr' shares
            chosen = share_list[:thr]
            xs = [(c[0]) for c in chosen]
            ys = [(c[1]) for c in chosen]
            # reconstruct integer and convert to bytes
            try:
                # use shamir reconstruct which expects (i, share) pairs
                pairs = [(int(x), int(y)) for x, y in zip(xs, ys)]
                secret_len = SEED_BYTE_LEN
                seed_bytes = shamir.reconstruct_secret_bytes(pairs, secret_len)
                reconstructed[target] = seed_bytes
                logger.secure_log("info", "Reconstructed missing seed", target=target)
            except Exception as e:
                logger.secure_log("warning", "Failed to reconstruct seed", target=target, err=str(e))
                continue
        return reconstructed

    def compute_aggregate(self) -> Dict[str, Any]:
        """
        After masked updates are available and missing seeds reconstructed (if any),
        compute the final unmasked aggregate.

        Steps:
        1. Sum all masked updates (numpy arrays) across senders that sent.
        2. For each pair (i,j) with i sent and j sent or reconstructed, compute pairwise mask contributions
           and subtract/add appropriately to remove masking.
        3. Return aggregated plain parameters as numpy arrays.
        """
        # 1. collect param names
        if not self.masked_updates:
            raise RuntimeError("No masked updates to aggregate")
        param_names = list(next(iter(self.masked_updates.values())).keys())

        # Initialize aggregator
        agg = {p: None for p in param_names}
        # sum masked updates
        for sender, params in self.masked_updates.items():
            for p, arr in params.items():
                a = np.array(arr, dtype=np.float64)
                if agg[p] is None:
                    agg[p] = a.copy()
                else:
                    agg[p] += a

        # we need to remove pairwise masks. For every pair (i,j), client i added mask(i,j), client j added mask(j,i) = -mask(i,j)
        # if both i and j sent, their masks cancel; if one dropped, we must subtract mask contributed by live client(s) for that missing client.
        missing = self.missing_clients()
        reconstructed_seeds = self.reconstruct_missing_seeds() if missing else {}

        # For every missing client m, reconstruct its pairwise masks with each alive client and remove
        alive = [c for c in self.clients.keys() if c not in missing]
        for m in missing:
            seed_m = reconstructed_seeds.get(m)
            if seed_m is None:
                logger.secure_log("warning", "Cannot reconstruct missing client's seed; aggregation may be impossible", missing_client=m)
                continue
            # derive mask that missing client would have applied with each alive client (mask_manager uses pairwise derivation)
            for a in alive:
                # pairwise seed between missing m and alive a:
                pair_seed = mask_manager.derive_pairwise_seed(m, a, seed_m)
                # mask that missing would have added (and alive added opposite)
                masks = mask_manager.compute_mask_from_seed(pair_seed, self.param_shapes)
                # missing client added +masks, alive added -masks; since missing didn't send, we need to subtract the missing's contribution
                # thus we subtract masks (i.e., agg - masks)
                for p, arr in masks.items():
                    if p in agg and agg[p] is not None:
                        agg[p] -= arr  # remove missing client's mask contribution

        # convert back to float32 and return
        final = {p: agg[p].astype(np.float32) for p in param_names}
        return final


class Client:
    def __init__(self, client_id: str, server: Server, param_shapes: Dict[str, Tuple[int, ...]]):
        self.client_id = str(client_id)
        self.server = server
        self.param_shapes = param_shapes
        # create a per-client master seed (secret) used to derive pairwise seeds
        rng = dp_rng.get_numpy_rng()
        self.master_seed = rng.bytes(constants.DEFAULTS.get("SEED_BYTE_LEN", 32))
        # to be populated after registration: map client->(idx assigned by server for shares)
        self.peer_indices: Dict[str, int] = {}
        # store shares received for others' seeds
        self.received_shares_for_me: List[Tuple[int, int]] = []
        self.received_shares_of_others: Dict[str, List[Tuple[int, int]]] = {}

        # Register with server
        self.server.register_client(msg.RegisterClient(client_id=self.client_id, param_shapes=self.param_shapes), self)

    def prepare_and_send_shares(self):
        """
        Split this client's master_seed via Shamir and send shares to all other clients via server.relay_share.
        Threshold computed from server.expected_threshold()
        """
        n = len(self.server.clients)
        t = self.server.expected_threshold()
        shares = shamir.split_secret_bytes(self.master_seed, n=n, t=t)
        # shares is list of (i, share_int) with length n; map index to recipient client id by order of server.clients keys
        client_ids = list(self.server.clients.keys())
        for (idx, share_val), recipient in zip(shares, client_ids):
            pkg = msg.SharePackage(sender=self.client_id, recipient=recipient, share=(idx, int(share_val)))
            self.server.relay_share(pkg)
        logger.secure_log("info", "Sent Shamir shares to peers", sender=self.client_id)

    def collect_my_shares(self):
        """
        Collect shares sent to this client and store for reconstructing others if needed.
        """
        shares = self.server.collect_shares_for_recipient(self.client_id)
        self.received_shares_for_me.extend(shares)
        logger.secure_log("info", "Collected shares addressed to me", me=self.client_id, count=len(shares))

    def provide_unmask_shares(self, missing_clients: List[str]) -> List[msg.UnmaskShare]:
        """
        For each missing client m, provide the Shamir share corresponding to m that this client holds.
        This client will look at previously received shares (which were given to it as part of other clients' split_secret)
        and for those that correspond to the missing client, send them to the server for reconstruction.
        Note: In this simulated model, the server stored shares with recipient==this client; to get the appropriate share
        for a missing target, we look through the shares this client received earlier, expecting that each client received
        one share per other client in registration order.
        """
        out: List[msg.UnmaskShare] = []
        # Build mapping: for each origin client that sent shares, there should be exactly one share addressed to this client
        # But our simple relay records only share tuples; so we rely on order: earlier prepare_and_send_shares was called by each client.
        # For a robust mapping, the SharePackage could include origin info; here server kept share.sender; we preserved it.
        # So to provide unmask shares we search server.received_shares for packages with recipient == self.client_id and sender == missing_client
        for m in missing_clients:
            # collect share in server.received_shares history for (sender=m, recipient=self.client_id)
            # note: server.received_shares holds remaining undelivered shares -- previously delivered shares were removed
            # However, during prepare_and_send_shares we appended to received_shares; when server.collect_shares_for_recipient() was called,
            # those shares were removed. To ensure we can still provide the share here, the server should have delivered them to the client.
            # In this simulation, a correct flow is:
            #   1) all clients call prepare_and_send_shares()
            #   2) all clients call collect_my_shares() to receive their shares (which will be stored in client's received_shares_for_me)
            # Thus here we simply look in self.received_shares_for_me for shares whose sender==m
            matching = []
            for (idx, share_val) in self.received_shares_for_me:
                # We don't know which missing client the share corresponds to because server.collect_shares_for_recipient removed sender info.
                # To maintain mapping, we instead rely on the fact that the shares were created in the same order of clients registration.
                # So a more robust approach is to reconstruct in server side using stored UnmaskShare objects that include sender info.
                # For simplicity, assume we kept mapping in self.received_shares_of_others populated earlier. If not present, skip.
                if m in self.received_shares_of_others:
                    for s in self.received_shares_of_others[m]:
                        out.append(msg.UnmaskShare(sender=self.client_id, missing_client=m, share=(s[0], s[1])))
                else:
                    # cannot provide share for this missing client
                    continue
        # If the simulation tracks shares differently, this function should be adapted.
        return out

    def send_masked_update(self, local_update: Dict[str, np.ndarray]):
        """
        Compute masked update and send to server:
        For each peer j: derive pairwise seed with j and compute mask; client adds masks where appropriate.

        For Bonawitz-style sign convention:
        - Each client i: add mask(i,j) for all j>i, subtract mask(i,j) for all j<i
        This ensures pairwise masks cancel when both participants are present.
        """
        # compute pairwise seeds between self and all peers using master_seed
        client_ids = list(self.server.clients.keys())
        my_index = client_ids.index(self.client_id)

        # Build masked params as numpy arrays
        masked = {}
        # start from original update (float64 for accumulation safety)
        for p, arr in local_update.items():
            masked[p] = np.array(arr, dtype=np.float64)

        for j, other in enumerate(client_ids):
            if other == self.client_id:
                continue
            pair_seed = mask_manager.derive_pairwise_seed(self.client_id, other, self.master_seed)
            masks = mask_manager.compute_mask_from_seed(pair_seed, self.param_shapes)
            # sign: if self.client_id < other -> add masks, else subtract masks
            if self.client_id < other:
                sign = +1.0
            else:
                sign = -1.0
            for p, arr in masks.items():
                masked[p] += sign * arr  # apply mask

        # send to server
        mu = msg.MaskedUpdate(sender=self.client_id, masked_params={k: v.astype(np.float32) for k, v in masked.items()})
        self.server.receive_masked_update(mu)
        logger.secure_log("info", "Sent masked update", sender=self.client_id)

    # helper to store mapping from origin -> (idx, share) when receiving initial shares
    def accept_initial_shares(self):
        """
        This method should be called by each client after server.relay_share was used to deliver SharePackages.
        It collects shares addressed to this client from server and stores them in a mapping keyed by origin client id.
        """
        # Collect raw share packages for this recipient
        pkgs = []
        # server.received_shares currently holds only undelivered shares; but earlier server.relay_share appended everything
        # So server.collect_shares_for_recipient will remove those packages and return share tuples WITHOUT sender info.
        # To keep sender info we instead parse server.received_shares directly.
        remaining = []
        for p in list(self.server.received_shares):
            if p.recipient == self.client_id:
                pkgs.append(p)
            else:
                remaining.append(p)
        # remove processed pkgs from server buffer
        self.server.received_shares = remaining

        # store mapping: sender -> share
        for pkg in pkgs:
            sender = pkg.sender
            share = pkg.share
            self.received_shares_of_others.setdefault(sender, []).append(share)
        # also build received_shares_for_me for reconstruction if needed (list of share tuples)
        self.received_shares_for_me = [pkg.share for pkg in pkgs]
        logger.secure_log("info", "Accepted initial shares", client=self.client_id, received=len(pkgs))
