import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc
import os
import pickle
from huggingface_hub import login
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Login to Hugging Face
login(token="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") 

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Load the Mistral model and tokenizer from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with low memory usage from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load the sentence transformer model for embeddings
embed_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Load the FAISS index
index_path = r"C:\Users\rajiv\OneDrive\Desktop\projecttt\rishika2rag" # Update this path
index = faiss.read_index(os.path.join(index_path, "index.faiss"))
print(f"FAISS index total vectors: {index.ntotal}")
print(f"FAISS index dimension: {index.d}")

# Load the documents
with open(os.path.join(index_path, "index.pkl"), "rb") as f:
    stored_docs = pickle.load(f)

class Conversation:
    def __init__(self):
        self.history = []
    
    def add_exchange(self, question, answer):
        self.history.append({"question": question, "answer": answer})
    
    def get_context(self, num_previous=2):
        context = ""
        for exchange in self.history[-num_previous:]:
            context += f"Previous Question: {exchange['question']}\n"
            context += f"Previous Answer: {exchange['answer']}\n\n"
        return context.strip()

def get_top_rag_answers(question, k=5):
    q_embedding = embed_model.encode([question])[0]
    D, I = index.search(np.array([q_embedding]).astype('float32'), k)
    if isinstance(stored_docs, dict):
        return [stored_docs.get(str(i), "Document found") for i in I[0]]
    elif isinstance(stored_docs, (list, tuple)):
        return [stored_docs[i] if i < len(stored_docs) else "Document not found" for i in I[0]]
    else:
        raise ValueError(f"Unexpected type for stored_docs: {type(stored_docs)}")

def answer_question(conversation, question):
    rag_answers = get_top_rag_answers(question)
    final_answer = combine_answers(conversation, question, rag_answers)
    conversation.add_exchange(question, final_answer)
    return final_answer

def combine_answers(conversation, question, rag_answers):
    combined_context = " ".join(rag_answers)
    conversation_context = conversation.get_context() if conversation.history else ""
    
    prompt = f"""You are a medical AI assistant designed to answer questions about health and medicine based on provided information. You must only respond to health and medicine-related queries. For any other topics, politely decline to answer.
Previous Conversation:
{conversation_context}
Context: {combined_context}
Question: {question}
Instructions:
1. If the question is related to health or medicine, provide a comprehensive and accurate answer based on the given context and previous conversation.
2. If the information in the context is insufficient to answer the question fully, say so and provide what information you can.
3. If the question is not related to health or medicine, respond with: "I'm sorry, but I can only answer questions related to health and medicine. This question appears to be outside my area of expertise."
4. Do not make up information. Stick to what is provided in the context and previous conversation.
5. You can answer basic questions like hi, hello etc.
6. Do not repeat the question or any part of the previous conversation in your answer.
7. Respond directly to the current question without mentioning previous exchanges.
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = response.split("Answer:")[-1].strip()
    
    del inputs, outputs
    clear_gpu_memory()
    
    return answer

# Initialize the conversation
conversation = Conversation()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    answer = answer_question(conversation, question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)