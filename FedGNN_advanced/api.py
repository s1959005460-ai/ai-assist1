# api.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
import asyncio, json, logging
from connection_manager import ConnectionManager
from models import get_model
from learner import Learner
from data_loader import synthetic_graph, split_non_iid_by_label
from server import AsyncFedServer
import torch

logger = logging.getLogger(__name__)
app = FastAPI()
manager = ConnectionManager()
GLOBAL_SERVER = None

@app.post("/start_training")
async def start_training(background_tasks: BackgroundTasks, config: dict = None):
    """
    Start a background federated training. Provide config JSON to override defaults.
    Example payload:
    {"rounds":5, "num_clients":4, "local_epochs":1, "use_delta": true}
    """
    global GLOBAL_SERVER
    cfg = config or {}
    rounds = int(cfg.get('rounds', 5))
    num_clients = int(cfg.get('num_clients', 4))
    local_epochs = int(cfg.get('local_epochs', 1))
    model_name = cfg.get('model', 'gcn')
    use_delta = bool(cfg.get('use_delta', True))
    out_dir = cfg.get('out_dir', './outputs')

    # create simple synthetic clients (replace with data loader if available)
    client_datas = [synthetic_graph(num_nodes=34, in_feats=16, num_classes=3) for _ in range(num_clients)]
    base = get_model(model_name, in_ch=16, hidden=32, out=3)
    clients = []
    for i in range(num_clients):
        m = get_model(model_name, in_ch=16, hidden=32, out=3)
        m.load_state_dict({k:v.clone() for k,v in base.state_dict().items()})
        clients.append(Learner(m, client_datas[i], lr=cfg.get('lr',0.05), local_epochs=local_epochs, device='cpu', topo_lambda=cfg.get('topo_lambda', 0.0)))

    GLOBAL_SERVER = AsyncFedServer(model_template=base, clients=clients, device='cpu', out_dir=out_dir, websocket_manager=manager, use_delta=use_delta)

    async def bg():
        try:
            await GLOBAL_SERVER.run_round(rounds=rounds)
        except Exception as e:
            logger.exception("Training crashed: %s", e)
            # notify clients
            try:
                await manager.send_message(json.dumps({'type':'error','payload': str(e)}))
            except Exception:
                pass

    # schedule background task
    background_tasks.add_task(lambda: asyncio.create_task(bg()))
    return {"status": "started", "rounds": rounds, "num_clients": num_clients}

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # keep alive and rely on manager.send_message to push updates
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
