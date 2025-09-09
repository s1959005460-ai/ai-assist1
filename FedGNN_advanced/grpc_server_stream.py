# grpc_server_stream.py
"""
gRPC server side example to receive streamed model bytes and reconstruct them.
Requires generated protos from protos/streaming.proto (use grpc_tools.protoc).
"""
import grpc
from concurrent import futures
import time
import logging
import io

# after protoc:
# from protos import streaming_pb2, streaming_pb2_grpc
# We'll guard imports to avoid runtime error if user hasn't generated pb2.
try:
    from protos import streaming_pb2, streaming_pb2_grpc
except Exception as e:
    streaming_pb2 = None
    streaming_pb2_grpc = None

from utils.serialization import deserialize_state_bytes

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
logging.basicConfig(level=logging.INFO)

class StreamServicer:  # streaming_pb2_grpc.StreamServiceServicer
    def UploadModel(self, request_iterator, context):
        # collect bytes
        buf = bytearray()
        key = None
        for chunk in request_iterator:
            if key is None:
                key = chunk.key
            buf.extend(chunk.data)
            if chunk.final_chunk:
                break
        try:
            state = deserialize_state_bytes(bytes(buf))
            # here you could save state or apply to server model
            logging.info("Received model key=%s, params=%d", key, len(state))
            return streaming_pb2.Ack(ok=True, msg="Received")
        except Exception as e:
            logging.exception("Failed to deserialize model: %s", e)
            return streaming_pb2.Ack(ok=False, msg=str(e))

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    if streaming_pb2_grpc:
        streaming_pb2_grpc.add_StreamServiceServicer_to_server(StreamServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC streaming server listening on {port}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve(50051)
