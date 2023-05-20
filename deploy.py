from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = load_model("model.h5")
ind = "index.html"

@app.route("/")
def home():
    ans = "Nothing to tell"
    return render_template(ind, **locals())


@app.route("/predict", methods=["POST", "GET"])
def predict():
    max_len = 150
    with open("tokenizer.pickle", "rb") as handle:
        tok = pickle.load(handle)
    sample_texts = request.form["text_input"]
    if sample_texts == "":
        ans = "Nothing to tell"
        return render_template(ind, **locals())
    txts = tok.texts_to_sequences([sample_texts])
    txts = pad_sequences(txts, maxlen=max_len)
    preds = model.predict(txts)
    if preds <= 0.2:
        ans = "Ham"
    else:
        ans = "Spam"
    return render_template(ind, **locals())


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,debug=False)
