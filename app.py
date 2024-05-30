from flask import Flask, request, jsonify
from keras.models import load_model
import backend.Kranos as Kranos

app = Flask(__name__)

# Load your pre-trained model and define necessary functions
model = load_model("model.h5")

@app.route('/api/data', methods=['GET'])
def get_data():
    user_input = request.json['message']

    # Process user input using your model and generate a response
    response = Kranos.predict_response(user_input)

    # Return the response as a JSON object
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
