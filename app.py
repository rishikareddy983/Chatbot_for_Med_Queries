import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc
import os
import pickle
from huggingface_hub import login
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
from pymongo import MongoClient
from bson import ObjectId
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:3000", "http://192.168.0.123:3000"]}})

login("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")

app.config['SECRET_KEY'] = '007'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['user_database']
users_collection = db['users']
chat_history_collection = db['chat_history']

login_manager = LoginManager(app)
login_manager.login_view = 'login'

device = "cuda" if torch.cuda.is_available() else "cpu"

# ... (keep all the model loading code and other functions as they were) ...
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

embed_model = SentenceTransformer('all-mpnet-base-v2', device=device)

index_path = r"C:\Users\rajiv\OneDrive\Desktop\projecttt\rishika2rag"
with open(os.path.join(index_path, "index.pkl"), "rb") as f:
    stored_docs = pickle.load(f)
faiss_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
embed_model = SentenceTransformer('all-mpnet-base-v2', device=device)

docstore, doc_mapping = stored_docs

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

class Conversation:
    def __init__(self):
        self.history = []
    
    def add_exchange(self, question, answer):
        self.history.append({"question": question, "answer": answer})
    
    def get_context(self, num_previous=2):
        context = ""
        for exchange in self.history[-num_previous:]:
            context += f"Previous Question: {exchange['question']}\nPrevious Answer: {exchange['answer']}\n\n"
        return context.strip()

conversations = {}

# ... (keep all the other functions as they were) ...

def get_top_rag_answers(question, k=5):
    q_embedding = embed_model.encode([question])[0]
    D, I = faiss_index.search(np.array([q_embedding]).astype('float32'), k)
    
    print("FAISS Index Results (IDs):", I[0])
    print("FAISS Distances:", D[0])
    
    doc_ids = [doc_mapping.get(i) for i in I[0] if i in doc_mapping]
    answers = [docstore._dict.get(doc_id, "Document not found") for doc_id in doc_ids]
    
    return [answer.page_content if hasattr(answer, 'page_content') else str(answer) for answer in answers]

def answer_question(user_id, question):
    if user_id not in conversations:
        conversations[user_id] = Conversation()
    conversation = conversations[user_id]
    rag_answers = get_top_rag_answers(question)
    # Print the top RAG answers to the terminal
    print(f"Top 5 RAG Answers for question '{question}':")
    for i, answer in enumerate(rag_answers, start=1):
        print(f"{i}. {answer[:200]}...")  # Print the first 200 characters of each answer
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
5. You can answer basic questions like hi, hello, how are you, who are you, etc.in not more than 30 words.
6. Do not repeat the question or any part of the previous conversation in your answer.
7. Respond directly to the current question without mentioning previous exchanges.
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = response.split("Answer:")[-1].strip()
    
    del inputs, outputs
    clear_gpu_memory()
    
    return answer

@app.route('/')
def index():
    return send_file('auth.html')

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            return jsonify({'message': 'Username or email already exists'}), 400

        password_hash = generate_password_hash(password)
        new_user = {
            'username': username,
            'email': email,
            'password_hash': password_hash
        }
        users_collection.insert_one(new_user)

        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        app.logger.error(f"Signup error: {str(e)}")
        return jsonify({'message': 'An error occurred during signup'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user_data = users_collection.find_one({'username': data.get('username')})
        if user_data and check_password_hash(user_data['password_hash'], data.get('password')):
            user = User(user_data)
            login_user(user)
            session['user_id'] = str(user.id)
            conversations[session['user_id']] = Conversation()
            return jsonify({'message': 'Logged in successfully'}), 200
        return jsonify({'message': 'Invalid username or password'}), 401
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'message': 'An error occurred during login'}), 500

@app.route('/logout')
@login_required
def logout():
    if 'user_id' in session:
        conversations.pop(session['user_id'], None)
        session.pop('user_id', None)
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    question = data.get('question')
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 400
    answer = answer_question(user_id, question)
    
    # Store chat history in MongoDB
    chat_entry = {
        'user_id': user_id,
        'question': question,
        'answer': answer,
        'timestamp': datetime.utcnow()
    }
    chat_history_collection.insert_one(chat_entry)
    
    return jsonify({"answer": answer})

@app.route('/chat_history', methods=['GET'])
@login_required
def get_chat_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session not found'}), 400
    
    chat_history = list(chat_history_collection.find({'user_id': user_id}, {'_id': 0, 'user_id': 0}))
    return jsonify(chat_history)

@app.route('/protected')
@login_required
def protected():
    return jsonify({'message': f'Hello, {current_user.username}! This is a protected route.'}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)