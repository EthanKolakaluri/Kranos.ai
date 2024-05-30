# explaination: This is the file that will be used to run the server
# The server will be used to communicate with the web app
# The web app will send a request to the server with the input text
# The server will pass the input text to the model and get a response
# The server will send the response back to the web app
# The web app will display the response to the user



from http.server import BaseHTTPRequestHandler, HTTPServer
import json
#from my_model import predict_response # replace with your own model function
import backend.Kranos as Kranos

# define the server's response handler
class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # get the request data
        content_length = int(self.headers['Content-Length'])
        request_data = self.rfile.read(content_length).decode('utf-8')
        
        # parse the request data as JSON
        request_json = json.loads(request_data)
        
        # get the input text from the request JSON
        input_text = request_json['input_text']
        
        # pass the input text to your model function to get a response
        response_text = Kranos.predict_response(input_text)
        
        # create a JSON response with the model's response
        response_json = {'response_text': response_text}
        response_data = json.dumps(response_json).encode('utf-8')
        
        # set the response headers and send the response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_data))
        self.end_headers()
        self.wfile.write(response_data)
        
# define the server's main function
def main():
    PORT = 8000 # set the server's port
    server_address = ('', PORT)
    
    # create and start the server
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting server on port %s...' % PORT)
    httpd.serve_forever()


# run the server
if __name__ == '__main__':
    main()
