from dataclasses import asdict

from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from flask_socketio import SocketIO

import modelResult
from model.QualityHGNN.train import Train_QHGNN
from model.MoonLabHGNN.train import Train_MoonLabHGNN
from model.AllSetTransformer.train import Train_AllSetTransformer
from parameters import InputType, SelectParameter, get_allset_parameters, get_moonlab_parameters, get_qhgnn_parameters, serialize
from datetime import datetime
import traceback

ALLOWED_ORIGINS = {"http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"}
ALLOWED_HOSTS = {"localhost:5002", "127.0.0.1:5002"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(
    app,
    cors_allowed_origins=list(ALLOWED_ORIGINS),
    async_mode="threading"
)
api = Api(app, version='1.0', title='HGCNN API',
          description='Hypergraph Neural Network Training API',
          doc='/docs')

# Track training jobs
training_jobs = {}

# Define models for Swagger documentation
train_params_model = api.model('TrainParameters', {
    'num_epochs': fields.Integer(default=1000, description='Number of training epochs'),
    'lr': fields.Float(default=0.001, description='Learning rate'),
    'hidden_layer_size': fields.Integer(default=128, description='Size of hidden layers'),
    'train_proportion': fields.Float(default=0.8, description='Proportion of data for training'),
    'valid_proportion': fields.Float(default=0.25, description='Proportion of data for validation (only for AllSetTransformer)'),
    'dropout': fields.Float(default=0.5, description='Dropout rate'),
    'weight_decay': fields.Float(default=5e-4, description='L2 regularization parameter'),
    'gamma': fields.Float(default=0.5, description='Learning rate decay factor'),
    'milestones_input': fields.String(default='50,100', description='Comma-separated epoch milestones for LR decay'),
    'seed': fields.Integer(default="Delete or set integer", description='Optinally set a seed for reproducibility'),
})

model_option_model = api.model('ModelOption', {
    'name': fields.String(description='Parameter name'),
    'options': fields.List(fields.String, description='Available model options'),
    'type': fields.String(description='Parameter type'),
})

options=["allset","moonlab", "qhgnn"]


@app.before_request
def restrict_to_localhost_frontend():
    host = request.host.split("@")[-1]
    if host not in ALLOWED_HOSTS:
        return {"error": "Forbidden host."}, 403

    origin = request.headers.get("Origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return {"error": "Forbidden origin. Only localhost:8000 is allowed."}, 403

@app.route("/")
def home():
    return "Hello, World! This is a Flask HTTP server."

@api.route("/models")
class Models(Resource):
    @api.marshal_with(model_option_model)
    def get(self):
        """Get available models"""
        return SelectParameter(name="Select Model", options=options, type=InputType.Select)

@api.route("/params/<model>")
class Parameters(Resource):
    def get(self, model: str):
        """Get parameters for a specific model"""
        match model:
            case "allset":
                return jsonify(serialize(get_allset_parameters()))
            case "moonlab":
                return jsonify(serialize(get_moonlab_parameters()))
            case "qhgnn":
                return jsonify(serialize(get_qhgnn_parameters()))
            case _:
                return {"error": "Model not found."}, 404

@api.route("/train/<model>/<job_id>")
class Train(Resource):
    @api.expect(train_params_model)
    def post(self, model: str, job_id: str):
        """Train a model with specified parameters (async)"""
        data = request.get_json() or {}

        if model not in options:
            return {"error": "Model not found."}, 404

        if model == "allset" and float(data.get("valid_proportion", 0.25)) + float(data.get("train_proportion", 0.5)) > 1.0:
            return {"error": "Train proportion and valid proportion must sum to 1 or less."}, 400

        int_fields = ["num_epochs", "hidden_layer_size", "seed"]
        float_fields = ["lr", "train_proportion", "valid_proportion", "dropout", "weight_decay", "gamma"]

        parsed_data = {}
        for k, v in data.items():
            if v == '':
                parsed_data[k] = None
            elif k in int_fields:
                parsed_data[k] = int(v)
            elif k in float_fields:
                parsed_data[k] = float(v)
            else:
                parsed_data[k] = v

        res = train_model_async(model, parsed_data, job_id)

        return {
            "status": "completed",
            "message": f"Training {model} model finished",
            "job_id": job_id,
            "result": res
        }, 200

def socket_logger(message, job_id=None, progress=None):
    socketio.emit(
        "job_update",
        {
            "job_id": job_id,
            "message": message,
            "progress": progress
        }
    )

    print(message)  # optional but useful

def train_model_async(model: str, data: dict, job_id: str):
    """Run model training in a background thread"""
    training_jobs[job_id] = {
        "status": "running",
        "model": model,
        "started_at": datetime.now().isoformat(),
        "message": f"Training {model} model..."
    }
    res = None
    try:
        match model:
            case "qhgnn":
                trainer = Train_QHGNN()
                res = trainer.train(
                    num_epochs=data.get("num_epochs", 1000),
                    lr=data.get("lr", 0.001),
                    hidden_layer_size=data.get("hidden_layer_size", 128),
                    train_proportion=data.get("train_proportion", 0.8),
                    dropout=data.get("dropout", 0.5),
                    weight_decay=data.get("weight_decay", 5e-4),
                    gamma=data.get("gamma", 0.5),
                    milestones_input=data.get("milestones_input", "50,100"),
                    seed=data.get("seed"),
                    job_id=job_id,
                    socket_logger=socket_logger
                )
            case "allset":
                trainer = Train_AllSetTransformer()
                res = trainer.train(
                    num_epochs=data.get("num_epochs", 1000),
                    lr=data.get("lr", 0.001),
                    hidden_layer_size=data.get("hidden_layer_size", 64),
                    train_proportion=data.get("train_proportion", 0.5),
                    valid_proportion=data.get("valid_proportion", 0.25),
                    dropout=data.get("dropout", 0.0),
                    weight_decay=data.get("weight_decay", 0.0),
                    seed=data.get("seed"),
                    job_id=job_id,
                    socket_logger=socket_logger
                )
            case "moonlab":
                trainer = Train_MoonLabHGNN()
                res = trainer.train(
                    num_epochs=data.get("num_epochs", 1000),
                    lr=data.get("lr", 0.001),
                    hidden_layer_size=data.get("hidden_layer_size", 128),
                    train_proportion=data.get("train_proportion", 0.8),
                    dropout=data.get("dropout", 0.5),
                    weight_decay=data.get("weight_decay", 5e-4),
                    gamma=data.get("gamma", 0.5),
                    milestones_input=data.get("milestones_input", "50,100"),
                    seed=data.get("seed"),
                    job_id=job_id,
                    socket_logger=socket_logger
                )

    except Exception as e:
        print("Oops")
        print(f"Error: {str(e)}")
        traceback.print_exc()
    return asdict(res)

@socketio.on("subscribe_job")
def handle_subscribe(data):
    job_id = data.get("job_id")
    if not job_id:
        return


@socketio.on("connect")
def handle_connect(auth=None):
    print("Client connected")

if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=5002, allow_unsafe_werkzeug=True)
