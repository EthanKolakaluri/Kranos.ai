from flask import Flask, request, jsonify
import mysql.connector

app = Flask(__name__)

# Configure database connection
db_config = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'database': 'your_database'
}

# Create a database connection
cnx = mysql.connector.connect(**db_config)

# Create a cursor object
cursor = cnx.cursor()

# Define a route to store a new conversation
@app.route('/conversations', methods=['POST'])
def create_conversation():
    data = request.get_json()
    user_id = data['user_id']
    chatbot_id = data['chatbot_id']
    conversation_start = data['conversation_start']

    # Insert conversation into database
    cursor.execute("INSERT INTO Conversations (user_id, chatbot_id, conversation_start) VALUES (%s, %s, %s)", (user_id, chatbot_id, conversation_start))
    cnx.commit()

    # Return the conversation ID
    conversation_id = cursor.lastrowid
    return jsonify({'conversation_id': conversation_id})

# Define a route to store a new message
@app.route('/messages', methods=['POST'])
def create_message():
    data = request.get_json()
    conversation_id = data['conversation_id']
    message = data['message']
    sender = data['sender']

    # Insert message into database
    cursor.execute("INSERT INTO Messages (conversation_id, message, sender) VALUES (%s, %s, %s)", (conversation_id, message, sender))
    cnx.commit()

    # Return the message ID
    message_id = cursor.lastrowid
    return jsonify({'message_id': message_id})

# Define a route to retrieve a conversation
@app.route('/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    cursor.execute("SELECT * FROM Conversations WHERE id = %s", (conversation_id,))
    conversation = cursor.fetchone()

    # Return the conversation data
    return jsonify({'conversation': conversation})

# Define a route to retrieve a list of messages in a conversation
@app.route('/conversations/<int:conversation_id>/messages', methods=['GET'])
def get_messages(conversation_id):
    cursor.execute("SELECT * FROM Messages WHERE conversation_id = %s", (conversation_id,))
    messages = cursor.fetchall()

    # Return the list of messages
    return jsonify({'messages': messages})

if __name__ == '__main__':
    app.run(debug=True)