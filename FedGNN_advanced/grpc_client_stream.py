# grpc_client_stream.py
"""
gRPC client example: stream serialized model state_dict to server using streaming RPC.
"""
import grpc
import io
import torch
from utils.serialization import serialize_state_dict, chunk_bytes

try:
    from protos import streaming_pb2, streaming_pb2_grpc
except Exception:
    streaming_pb2 = None
    streaming_pb2_grpc = None

def send_model_state(server_addr: str, state_dict, key='model-1', chunk_size=64*1024):
    data = serialize_state_dict(state_dict)
    channel = grpc.insecure_channel(server_addr)
    stub = streaming_pb2_grpc.StreamServiceStub(channel)
    def gen():
        for i, c in enumerate(chunk_bytes(data, chunk_size)):
            final = (i == (len(data) - 1) // chunk_size)
            yield streaming_pb2.Chunk(data=c, key=key, final_chunk=final)
    resp = stub.UploadModel(gen())
    return resp

# Usage example:
# resp = send_model_state('localhost:50051', my_state_dict)
# print(resp)
