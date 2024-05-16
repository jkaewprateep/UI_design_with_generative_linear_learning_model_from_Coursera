# UI design with generative linear learning model from Coursera
UI design with generative linear learning model from Coursera

* IBM AI Developer Professional [certification]( https://coursera.org/share/95fa5c2bf36ea52759dcabc50e1a81b0 )
* IBM Front-End Developer Professional [certification]( https://coursera.org/share/b6a39450002ea820b057a60286aa3356 )  

<p align="center" width="100%">
    <img width="34%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/Frontend%20instructor.png">
    <img width="34%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/AI%20instructor.png">
    <img width="12.77%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/09.jpg"> </br>
    <b> Front-End Developer | AI Developer | My favorite teacher 💃( 👩‍🏫 ) </b> </br>
    <b> Pictures from the Internet </b> </br>
</p>

## Glossary application design

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/Glossary%20application%20design.png"> </br>
    <b> Glossary application design </b> </br>
</p>

## Customer search application design

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/Customer%20serach%20application%20design.png"> </br>
    <b> Customer search application design </b> </br>
</p>

## LLM Chat application

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/LLM%20-%20chat%201.png">
    <img width="40%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/LLM%20-%20chat%202.png"> </br>
    <b> LLM Chat application </b> </br>
</p>

### Simple request-response

```
from sentiment_analysis import sentiment_analyzer
import json
import requests

response = sentiment_analyzer("🧸💬 There are 10 principles of DekDee ... ")

url = "https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
headers = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}
myobj = { "raw_document": { "text": "as987da-6s2d aweadsa" } }
response = requests.post(url, json = myobj, headers=headers)
print(response.status_code)

myobj = { "raw_document": { "text": "Testing this application for error handling" } }
response = requests.post(url, json = myobj, headers=headers)
print(response.status_code)
print(response)
```

### Sample response

```
>>
{'emotionPredictions': [{'emotion': {'anger': 0.010162572, 'disgust': 0.51078576, 'fear': 0.025222138, 
'joy': 0.77610445, 'sadness': 0.061564375}, 'target': '', 
'emotionMentions': [{'span': {'begin': 0, 'end': 40, 'text': '🧸💬 There are 10 principles of DekDee ...'}, 
'emotion': {'anger': 0.010162572, 'disgust': 0.51078576, 'fear': 0.025222138, 'joy': 0.77610445, 'sadness': 0.061564375}}]}],
'producerId': {'name': 'Ensemble Aggregated Emotion Workflow', 'version': '0.0.1'}}
```

## Logicals assignment application design

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/Logical%20assignment%20application%20design.png"> </br>
    <b> Logicals assignment application design </b> </br>
</p>

## NLP emotion detection application

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/NLP%20-%20emotion%20detection.png"> </br>
    <b> NLP emotion detection application </b> </br>
</p>

### Transformer
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Choosing a model
model_name = "meta-llama/Meta-Llama-Guard-2-8B";
# model_name = "facebook/blenderbot-400M-distill"

# Step 4: Fetch the model and initialize a tokenizer
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name);
tokenizer = AutoTokenizer.from_pretrained(model_name);

# Step 5.1: Keeping track of conversation history
conversation_history = [];

# Step 5.2: Encoding the conversation history
history_string = "\n".join(conversation_history);

# Step 5.3: Fetch prompt from user
input_text ="hello, how are you doing?"

# Step 5.4: Tokenization of user prompt and chat history
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

# Step 5.5: Generate output from the model
outputs = model.generate(**inputs)
print(outputs)

# Step 5.6: Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

# Step 5.7: Update conversation history
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)

# Step 6: Repeat
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
```

## NLP image to text application

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/NLP%20image%20to%20text.png"> </br>
    <b> NLP image to text application </b> </br>
</p>

### Server

```
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    text = speech_to_text(audio_binary) # Call speech_to_text function to transcribe the speech

    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    print(response.data)
    return response
```

## NLP image object detection

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/image_region_detection.png"> </br>
    <b> NLP image object detection </b> </br>
</p>

## Voice assistance

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/Voice%20assistance.png"> </br>
    <b> Voice assistance </b> </br>
</p>

### App.js

```
from flask import Flask, render_template            # newly added
from flask_cors import CORS                         # newly added

from transformers import AutoModelForSeq2SeqLM      # newly added
from transformers import AutoTokenizer              # newly added

from flask import request                           # newly added
import json                                         # newly added

"""""""""""""""""""""""""""""""""""""""""""""""""""""
MODEL DEFINED
"""""""""""""""""""""""""""""""""""""""""""""""""""""
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

"""""""""""""""""""""""""""""""""""""""""""""""""""""
EXPECTED MESSAGE
"""""""""""""""""""""""""""""""""""""""""""""""""""""
expected_message = {
    'prompt': 'message'
}

app = Flask(__name__)
CORS(app);                                          # newly added
```

### App.js - routes banana

```
@app.route('/bananas')
def bananas():
    return '🍌 This page has bananas!'
```

### App.js - routes chatbots

```
@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # Create conversation history string
    history = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length= 60)  # max_length will acuse model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response
```

## Sign-up form application design

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/UI_design_with_generative_linear_learning_model_from_Coursera/blob/main/signup%20form%20application%20design.png"> </br>
    <b> Sign-up form application design </b> </br>
</p>
