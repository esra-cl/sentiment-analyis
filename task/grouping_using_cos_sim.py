import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

class grouping_:
    def __init__(self, topic=None, opinion=None):
        self.topic = topic
        self.opinion = opinion
        self.op = None
        self.top = None
        self.op_path = r"C:\Users\HP\Downloads\task\dataset\opinions.csv"
        self.top_path = r"C:\Users\HP\Downloads\task\dataset\topics.csv"
        self.dataset_df, self.group_df = None, None
        self.topics, self.opinions = None, None
        self.opinions_embeddings, self.topics_embeddings = None, None
        self.st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.runner()

    def runner(self):
        self.load_files()
        self.create_dataset()
        self.opinions = pd.DataFrame(self.dataset_df[['opinions', 'type']], columns=['opinions', 'type'])
        self.topics = self.dataset_df[['topics']].to_numpy().flatten()
        self.load_update_Dogrouping()

    def load_files(self):
        self.op = pd.read_csv(self.op_path)
        self.top = pd.read_csv(self.top_path)
        return self.op, self.top

    def create_dataset(self):
        merged_df = pd.merge(
            self.top[['topic_id', 'text']],
            self.op[['topic_id', 'text', 'type']],
            on='topic_id',
            suffixes=('_topic', '_opinion')
        )
        merged_df = merged_df.rename(columns={
            'text_topic': 'topics',
            'text_opinion': 'opinions'
        })
        self.dataset_df = merged_df[['topics', 'opinions', 'type']]
        self.dataset_df.to_csv('merged_dataset.csv', index=False)
        return self.dataset_df

    def embedding_data(self):
        opinions_texts = self.opinions['opinions'].tolist()
        print("Embedding opinions...")
        self.opinions_embeddings = self.st_model.encode(opinions_texts, batch_size=64, convert_to_tensor=True)
        self.topics_embeddings = self.st_model.encode(self.topics, batch_size=64, convert_to_tensor=True)
        np.save(r'C:\Users\HP\Downloads\task\embadded_\opinion_embeddings.npy', self.opinions_embeddings.cpu().numpy())
        np.save(r'C:\Users\HP\Downloads\task\embadded_\topics_embeddings.npy', self.topics_embeddings.cpu().numpy())
        print("Embedding complete.")

    def embedding_data2(self, sentiment):
        embedded_sentiment = self.st_model.encode(sentiment, convert_to_tensor=True)
        return embedded_sentiment

    def load_update_Dogrouping(self):
        self.opinions_embeddings = torch.tensor(np.load(r'C:\Users\HP\Downloads\task\embadded_\opinion_embeddings.npy'), dtype=torch.float32)
        self.topics_embeddings = torch.tensor(np.load(r'C:\Users\HP\Downloads\task\embadded_\topics_embeddings.npy'), dtype=torch.float32)
        
        if self.topic is not None:
            embedded_topic = self.embedding_data2(self.topic)
            self.topic = embedded_topic
            if not any(torch.equal(embedded_topic, te) for te in self.topics_embeddings):
                self.topics_embeddings = torch.vstack([self.topics_embeddings, embedded_topic])
        
        if self.opinion is not None:
            embedded_opinion = self.embedding_data2(self.opinion)
            self.opinion = embedded_opinion
            if not any(torch.equal(embedded_opinion, oe) for oe in self.opinions_embeddings):
                self.opinions_embeddings = torch.vstack([self.opinions_embeddings, embedded_opinion])

        return self.opinions_embeddings, self.topics_embeddings

    def group_(self):
        opinions_list = []
        if self.topic is not None:
            similarity_scores = util.pytorch_cos_sim(self.opinions_embeddings, self.topic)
            similarity_scores = similarity_scores.flatten().tolist()
            opinions_with_scores = list(zip(self.opinions['opinions'], similarity_scores))
            for opinion_text, score in opinions_with_scores:
                if score > 0.55:
                    opinions_list.append(opinion_text)
        return opinions_list

    def group_for_train(self, threshold=0.40):
        print(f"Grouping with threshold={threshold}...")
        similarity_scores = util.pytorch_cos_sim(self.opinions_embeddings, self.topics_embeddings)
        best_matches = similarity_scores.argmax(dim=1)
        grouped_data = []

        for i, match in enumerate(best_matches):
            if similarity_scores[i, match] >= threshold:
                grouped_data.append({
                    'topic_text': self.topics[match.item()],
                    'opinion_text': self.opinions.iloc[i, 0],
                    'label': self.opinions.iloc[i, 1],
                })

        self.group_df = pd.DataFrame(grouped_data)
        self.group_df.to_csv('grouped_data.csv', index=True)
        print("Grouping done.")

    def evaluation_(self):
        merged_opinions_topics = pd.merge(
            self.top[['topic_id', 'text']],
            self.op[['topic_id', 'text', 'type']],
            on='topic_id',
            suffixes=('_topic', '_opinion')
        )
        
        merged_opinions_topics = merged_opinions_topics.rename(columns={
            'text_topic': 'topics',
            'text_opinion': 'opinions'
        })

        comparison_df = pd.merge(
            self.group_df[['topic_text', 'opinion_text']],
            merged_opinions_topics[['opinions', 'topics']],
            left_on=['opinion_text', 'topic_text'],
            right_on=['opinions', 'topics'],
            how='inner'
        )

        mismatches = comparison_df[comparison_df.isnull().any(axis=1)]
        total_mismatches = mismatches.shape[0]
        total_rows = self.group_df.shape[0]
        accuracy = (total_rows - total_mismatches) / total_rows

        print(f"Total Mismatches: {total_mismatches}")
        print(f"Total Rows: {total_rows}")
        print(f"Accuracy: {accuracy:.4f}")
        return total_mismatches, total_rows, accuracy

