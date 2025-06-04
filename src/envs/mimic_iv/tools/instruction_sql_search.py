import os
import json
import faiss

from typing import Any, Dict, List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE_DATA_DIR = 'src/envs/mimic_iv'
MIMIC_TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'mimic_train_data.json')
MIMIC_TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'mimic_train_label.json')
MIMIC_VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'mimic_valid_data.json')
MIMIC_VALID_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'mimic_valid_label.json')

class InstructionSQLSearch(BaseModel):
    data: List[Dict]
    model: Any
    index: Any
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self):
        with open(os.path.join(MIMIC_TRAIN_DATA_PATH), 'r') as f:
            mimic_train_data = json.load(f)
            
        with open(os.path.join(MIMIC_TRAIN_LABEL_PATH), 'r') as f:
            mimic_train_label = json.load(f)
            
        with open(os.path.join(MIMIC_VALID_DATA_PATH), 'r') as f:
            mimic_valid_data = json.load(f)
            
        with open(os.path.join(MIMIC_VALID_LABEL_PATH), 'r') as f:
            mimic_valid_label = json.load(f)
            
        total_data = mimic_train_data['data'] + mimic_valid_data['data']
        total_label = {}
        total_label.update(mimic_train_label)
        total_label.update(mimic_valid_label)
        
        data = []
        for sample in total_data:
            sample_id = sample['id']
            sample_question = sample['question']
            sample_label = total_label[sample_id]
            if sample_label == "null":
                continue
            sample_dict = {"id": sample_id, "question": sample_question, "label": sample_label}
            data.append(sample_dict)
        
        # print(len(data))
        # print(data[:5])
            
        # Extract the questions from the training data for embedding.
        questions = [sample['question'] for sample in data]
        
        # Initialize the sentence transformer model.
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Compute embeddings for each training question.
        corpus_embeddings = model.encode(questions, convert_to_numpy=True)
        
        # Build a FAISS index using L2 similarity.
        embedding_dim = corpus_embeddings.shape[1]
        
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(corpus_embeddings)  # Add the embeddings to the index.
        
        super().__init__(data=data, model=model, index=index)
    
    def invoke(self, instruction: str, k: int = 10) -> str:
        # Compute embedding for the new user query.
        query_embedding = self.model.encode([instruction], convert_to_numpy=True)
        
        # Search the FAISS index for the most similar questions.
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            # Get the corresponding training example.
            data_entry = self.data[idx]
            
            # Append the combined result.
            results.append(data_entry)

        answer = ""
        for r in results:
            answer += f"NLQ:{r['question']}\nSQL:{r['label']}\n"
        return answer

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "instruction_sql_search",
                "description": "Retrieve up to k instruction-SQL pairs that are relevant to the given user instruction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "The user instruction to search for."
                        },
                        "k": {
                            "type": "integer",
                            "description": "The maximum number of values to return. Default is 10."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

if __name__ == "__main__":
    instruction_sql_search = InstructionSQLSearch()
    print("Initialized")
    print(len(instruction_sql_search.data))
    print(instruction_sql_search.data[:5])
    
    print(instruction_sql_search.invoke('What are the ways to consume sodium bicarbonate?'))
    