from flask import Flask, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = '007'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['user_database']
users_collection = db['users']

login_manager = LoginManager(app)
login_manager.login_view = 'login'

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

@app.route('/signup', methods=['POST'])
def signup():
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

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user_data = users_collection.find_one({'username': data.get('username')})
    if user_data and check_password_hash(user_data['password_hash'], data.get('password')):
        user = User(user_data)
        login_user(user)
        return jsonify({'message': 'Logged in successfully'}), 200
    return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/protected')
@login_required
def protected():
    return jsonify({'message': f'Hello, {current_user.username}! This is a protected route.'}), 200

if __name__ == '__main__':
    app.run(debug=True)