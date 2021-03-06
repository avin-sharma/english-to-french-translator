
# # Keras + Flask for REST api


import pickle
import numpy as np
import flask
import io
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
english_tokenizer = None
french_tokenizer = None
y_id_to_word = {}


# ## Loading our model and tokenizer

def load_translation_model():
    # load the pre-trained Keras model 
    global model
    global english_tokenizer
    global french_tokenizer
    global y_id_to_word
    model = load_model('translation.h5')
    # loading
    with open('english_tokenizer.pickle', 'rb') as handle:
        english_tokenizer = pickle.load(handle)
    with open('french_tokenizer.pickle', 'rb') as handle:
        french_tokenizer = pickle.load(handle)
    y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
    y_id_to_word[0] = '<PAD>'


# ## Preprocessing the text for input to the model

def prepare_text(sentence):
    sentence = [english_tokenizer.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=21, padding='post')
    sentence = np.array(sentence)
    return sentence


# ## Converting numbers back to words


def logits_to_text(logits, tokenizer = french_tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = y_id_to_word

    return ' '.join([index_to_words[np.argmax(x)] for x in logits])


@app.route("/predict")
def predict():
    data = {"success": False}
    sentence = flask.request.args.get('sentence')
    sentence = prepare_text(sentence)
    sentence = model.predict(sentence)
    # sentence = logits_to_text(sentence)
    sentence = ' '.join([y_id_to_word[np.argmax(x)] for x in sentence[0]])
    sentence =sentence.replace(' <PAD>', '')
    data["success"] = True
    data["prediction"] = sentence
    # return the data dictionary as a JSON response
    return flask.jsonify(data)
    
print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
load_translation_model()


@app.route('/', methods=['GET'])
def home():
    vocab = ' '.join([key for key, value in english_tokenizer.word_index.items()])
    return "<h1>Vocabulary</h1>" + vocab
