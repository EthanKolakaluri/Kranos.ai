# explain: This file is the main file for the chatbot. It loads the intents file, 
# tokenizes the words, creates a dictionary, and trains the model.
#
# The model is a neural network with 3 layers. The first layer has 8 neurons,
# the second layer has 8 neurons, and the output layer has the same number of neurons
# as the number of tags. The model is trained for 1000 epochs.
#
# The model is saved as model.h5 and the words, labels, training, and output are saved
# as data.pickle. This is done so that the model doesn't have to be trained every time
# the chatbot is run.
#
# The chat function takes in the user input and returns the response from the model.
# The bag_of_words function tokenizes the user input and creates a bag of words array.
# The chat function uses the model to predict the tag of the user input and then returns
# a random response from the list of responses in the intents file.

import pickle
import tensorflow as tf
import numpy as np
import json
import random
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer    
import nltk
import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Example: Extracting all the paragraphs from the webpage
        paragraphs = soup.find_all('p')
        
        # Extract text from paragraphs and concatenate
        scraped_text = ""
        for paragraph in paragraphs:
            scraped_text += paragraph.text + " "
        
        return scraped_text
    else:
        # If the request was not successful, return an error message
        return "Error: Unable to retrieve webpage."

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]  # Remove punctuation and convert to lowercase
    return tokens

def extract_relevant_response(source_text, matching_tokens):
    # Example logic to extract a sentence containing the matching tokens
    sentences = nltk.sent_tokenize(source_text)
    for sentence in sentences:
        sentence_tokens = tokenize_text(sentence)
        if set(matching_tokens).issubset(set(sentence_tokens)):
            return sentence
    # If no relevant sentence found, return an empty string or handle it accordingly
    return ""

def replaced_responses(intent, source_urls):
    
    source_text = ""

    for url in source_urls:
        source_text += scrape_webpage(url) + " "

    # Tokenize and preprocess source text
    source_tokens = tokenize_text(source_text)

    # Match tokens with patterns in intent
    matching_tokens = set(intent['patterns']) & set(source_tokens)

    # If matching tokens found, concatenate them as the response
    if matching_tokens:
        intent['responses'] = extract_relevant_response(source_text, matching_tokens)
    else:
        intent['responses'] = random.choice(intent['responses']) 

    return intent['responses']

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def predict_response():
    
    source_urls = ["https://www.ibm.com/topics/artificial-intelligence", "https://weather.com/weather/hourbyhour/l/4bc782df8b53ef6bf89863fb91b82e5c2257063893829110d463f9c4a6062d4e"]

    print("Start talking with the bot! (type quit if you get bored)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                #responses = tg['responses']

                # Replace responses in intents with those from the source
                responses = replaced_responses(tg, source_urls)

        if "+" in inp.lower() or "*" in inp.lower() or "-" in inp.lower() or "/" in inp.lower() or "^" in inp.lower():
            
            # Extract the numbers and operator from the question
            num1, num2, operator = extract_numbers_and_operator(inp.lower())

            # Perform the calculation based on the operator
            if operator == "+":
                result = add(num1, num2)
            elif operator == "-":
                result = subtract(num1, num2)
            elif operator == "*":
                result = multiply(num1, num2)
            elif operator == "/":
                result = divide(num1, num2)
            elif operator == "^":
                result = power(num1,num2)

            print("The answer is " + str(result))
            #return "The answer is " + str(result)
        else:
             
            print(responses)    
            #return response
    
# Define a function to add two numbers
def add(x, y):
    return x + y

# Define a function to subtract two numbers
def subtract(x, y):
    return x - y

# Define a function to multiply two numbers
def multiply(x, y):
    return x * y

# Define a function to divide two numbers
def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def power(x, y):
    
    x1 = 1.0
    for i in range(int(y)):
        x1 *= x

    return x1

# Define a function to extract the numbers and operator from a math question
def extract_numbers_and_operator(question):

    # Look for the first two numbers and the operator
    num1 = None
    num2 = None
    operator = None
    
    if "." in question:
        words = question.split()
         
        for i, word in enumerate(words):
            
            if type(float(word)) == float and num1 is None:
                num1 = float(word)
            elif type(float(word)) == float and num2 is None:
                num2 = float(word)
            elif word in ["+", "-", "*", "/", "^"] and operator is None:
             operator = word

        # Return the numbers and operator as a tuple
        return num1, num2, operator
    else:
        for i, word in enumerate(question):
            
            if word.isnumeric() and num1 is None:
                num1 = float(word)
            elif word.isnumeric() and num2 is None:
                num2 = float(word)
            elif word in ["+", "-", "*", "/", "^"] and operator is None:
                operator = word

        # Return the numbers and operator as a tuple
        return num1, num2, operator

stemmer = LancasterStemmer()
response = None
count = 0

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

try: 
    # Load the data from the pickle file
    with open("data.pickle", "rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    # Tokenize the words and create a dictionary
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Tokenize the words and create a dictionary
    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    
    # Stem and sort the words
    words = [stemmer.stem(w.lower()) for w in words if w not in '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # Create training data
    training = []
    output = []

    # Create an array for each word in the bag of words
    out_empty = [0] * len(labels)

    for i, doc in enumerate(docs_x):
        bag = []

        stemmed_words = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in stemmed_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output), f)

try:
    model = tf.keras.models.load_model('model.h5')

    predict_response()
    
except:
    
    # Define the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(9, input_shape=(len(training[0]),)),
    tf.keras.layers.Dense(9),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(training, output, epochs=1200, batch_size=16)
              

    # Save the model
    model.save('model.h5')







