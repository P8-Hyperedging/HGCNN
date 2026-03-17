from flask import Flask, jsonify
from parameters import get_parameters, serialize

app = Flask(__name__)

@app.route("/")
def home():
    #Maybe run train?
    return "Hello, World! This is a Flask HTTP server."

@app.route("/about")
def about():
    return "This is the about page."

@app.route("/params")
def params():
    return jsonify(serialize(get_parameters()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
