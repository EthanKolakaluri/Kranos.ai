from flask import Flask, jsonify, request 
import Ebot2

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    
    data = {'message': 'Data from the Ethan''s backend'}
    user_input = request.json[data]

    # Process user input using your model and generate a response
    response = Ebot2.predict_response(user_input)

    # Return the response as a JSON object
    return jsonify({'response': response})

@app.route('file:///Users/beulahchaise/ethan/frontend/src/App.js')
def proxy(url):
    response1 = request(method=request.method, url=url, headers = {key: value for (key, value) in request.headers if key != 'Host'}, data = request.get_data(), cookies = request.cookies, allow_redirects = False)
    response1.headers.add('Access-Control-Allow_Origin', '*')
    response1.headers.add("Access-Control-Allow-Headers", 'Content Type')

    return jsonify(response1.json())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)

