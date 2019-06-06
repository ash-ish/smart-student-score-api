import flask
from datetime import datetime
import pytz
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

app = flask.Flask(__name__)
with open('classifier.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('classifier.h5')
graph = tf.get_default_graph()

def init():
    global model,graph
    # load the pre-trained Keras model
    with open('classifier.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('classifier.h5')
    graph = tf.get_default_graph()

@app.route("/predict", methods=["GET","POST"])
def predict():
    parameters = []
    parameters.append(flask.request.args.get('attendance'))
    parameters.append(flask.request.args.get('participation'))
    parameters.append(flask.request.args.get('marks'))
    parameters.append(flask.request.args.get('hackathon'))
    parameters.append(flask.request.args.get('certificate'))
    if parameters[0] :
        inputFeature = np.asarray(parameters).reshape(1, 3)
        print(inputFeature)
        with graph.as_default():
            raw_prediction = model.predict(inputFeature)[0][0]
        data = {"score": str(raw_prediction)}
    else:
        data = {"score": "0"}
    return flask.jsonify(data)  

if __name__ == '__main__':
    init()
    app.run(debug=True, use_reloader=True)
