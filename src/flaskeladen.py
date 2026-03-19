from flask import Flask, jsonify
from parameters import InputType, SelectParameter, get_allset_parameters, get_moonlab_parameters, get_qhgnn_parameters, get_parameters, serialize

app = Flask(__name__)

@app.route("/")
def home():
    #Maybe run train?
    return "Hello, World! This is a Flask HTTP server."

@app.route("/about")
def about():
    return "This is the about page."

@app.route("/params/<model>")
def params(model: str):
    match model:
        case "allset":
            return jsonify(serialize(get_allset_parameters()))
        case "moonlab":
            return jsonify(serialize(get_moonlab_parameters()))
        case "qhgnn":
            return jsonify(serialize(get_qhgnn_parameters()))
        case _:
            return jsonify(serialize(get_parameters()))

@app.route("/models")
def models():
    return jsonify(SelectParameter(name="Select Model", options=["allset","moonlab", "qhgnn"], type=InputType.Select))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
