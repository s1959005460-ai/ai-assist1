# grpc_server.py
"""
gRPC server skeleton that plugs into your server logic.
Run: python grpc_server.py
Note: you must `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/federated.proto` to generate pb2 files.
"""
import grpc
from concurrent import futures
import time
import torch
import io

# import generated pb2 / pb2_grpc after running protoc
# from protos import federated_pb2, federated_pb2_grpc
import logging
logging.basicConfig(level=logging.INFO)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class FederatedServicer:  # inherit federated_pb2_grpc.FederatedServiceServicer
    def __init__(self, server_obj):
        self.server_obj = server_obj

    def SendMaskedDelta(self, request, context):
        # deserialize training request and pass to server
        # validation, anti-replay, signature verification should be done here or in server_obj
        return None

def serve(server_obj, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # federated_pb2_grpc.add_FederatedServiceServicer_to_server(FederatedServicer(server_obj), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
