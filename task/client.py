import grpc
import proto.event_pb2_grpc as event_pb2_grpc
import proto.event_pb2 as event_pb2
import json

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = event_pb2_grpc.EventServiceStub(channel)
    
    # Manually input event details for a new topic
    event_id = input("Enter Event ID: ")
    event_type = "new_topic"
    topic_id = input("Enter Topic ID: ")
    topic_name = input("Enter Topic Name: ")
    topic_text = input("Enter Topic text: ")
    timestamp = input("Enter Event Timestamp: ")
    
    # Create the event payload
    payload = {
        "event_type": event_type,
        "data": {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "description": topic_text,
            "created_at": timestamp
        }
    }
    
    # Convert the payload to JSON string
    payload_json = json.dumps(payload)
    
    # Create the event
    event = event_pb2.Event(
        event_id=event_id,
        event_type=event_type,
        payload=payload_json,
        timestamp=timestamp
    )
    
    # Send the event
    response = stub.SendEvent(event)
    print("Response from server:", response.status)

if __name__ == "__main__":
    run()
