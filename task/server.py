import grpc
from concurrent import futures
import time
import proto.event_pb2_grpc as event_pb2_grpc
import proto.event_pb2 as event_pb2
import json
from application import app  # Assuming 'app' is the class you want to instantiate

from application import app  # Assuming 'app' class is defined in 'application.py'

class EventServiceServicer(event_pb2_grpc.EventServiceServicer):
    def SendEvent(self, request, context):
        print(f"Received Event: {request.event_type} with ID: {request.event_id}")
        
        if request.event_type == "new_topic":
            payload = json.loads(request.payload)
            topic_name = payload["data"]["topic_name"]
            description = payload["data"]["description"]
            print(f"New Topic Received: {topic_name}")
            
            # Use the app class to process the topic
            ob = app()
            processed_data = ob.compute_(description)
        else:
            processed_data = f"Processed data for event: {request.event_id}"
        
        return event_pb2.EventResponse(status=processed_data)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    event_pb2_grpc.add_EventServiceServicer_to_server(EventServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Server running on port 50051")
    server.start()
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
