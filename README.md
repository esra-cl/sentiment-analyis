# sentiment-analysiz

## Project Directory Overview
In the project directory, you will find the following:

Colab Directory: This directory contains the files mlm_model.ipynb and t5_llm_model.ipynb. These notebooks were used to fine-tune, save, and evaluate the models on the Colab platform. You can also find the evaluation metrics within these files.

Embeddings Directory: This directory includes topics_embeddings.npy and opinion_embeddings.npy, which are the files where the embeddings for topics and opinions are stored. These embeddings were generated to be used in the grouping process.

Models Directory: The mlm_model and t5_llm_model files in this directory contain the saved models, which will be used for classification and summarization tasks.

Proto Directory: This directory contains the .proto file, event_pb2.py, and event_pb2_grpc.py files, which define the structure of the gRPC API.

Grouping Module (grouping_using_cos_sim.py): This script is used to group topics with their respective opinions based on cosine similarity.

Application Module (application.py): This is the main application script that integrates all models and modules to process event messages.

Server (server.py): This script is responsible for receiving the event, processing the topic, and returning the results to the client.

Client (client.py): This script sends events to the server for processing.

## Processing the Topic Text
When the server receives a text to be processed, it calls the compute method from the application module. The compute method performs the following tasks:

Grouping Opinions: It identifies the opinions related to the topic using methods from the grouping module. The grouping is done with a threshold of 0.55.

Classifying Opinions: The grouped opinions are tokenized and sent to the RoBERTa model to classify their types (e.g., claim, counterclaim, etc.).

Generating the Conclusion: Finally, the topic is passed to the T5 (Text-to-Text) model to generate a conclusion for the topic.


## How to Run the Project
-> Open Two Terminals or PowerShell Windows:
    -> Rename one terminal as "Server" and the other as "Client."
-> Run the Server and Client Scripts:

    ->Execute the server.py file in the "Server" terminal.
    ->Execute the client.py file in the "Client" terminal.
    
The client will prompt you to enter the details 
of the topic event. Wait for the server to process the topic text and return the event message.

***NOT! 
 run the .ipynb files to save the models and use it . 
