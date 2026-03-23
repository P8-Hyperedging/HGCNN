from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields, Namespace
from model.QualityHGNN.train import Train_QHGNN
from parameters import InputType, SelectParameter, get_allset_parameters, get_moonlab_parameters, get_qhgnn_parameters, get_parameters, serialize
import threading
from datetime import datetime
import traceback

app = Flask(__name__)
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
    'dropout': fields.Float(default=0.5, description='Dropout rate'),
    'weight_decay': fields.Float(default=5e-4, description='L2 regularization parameter'),
    'gamma': fields.Float(default=0.5, description='Learning rate decay factor'),
    'milestones_input': fields.String(default='50,100', description='Comma-separated epoch milestones for LR decay'),
})

model_option_model = api.model('ModelOption', {
    'name': fields.String(description='Parameter name'),
    'options': fields.List(fields.String, description='Available model options'),
    'type': fields.String(description='Parameter type'),
})

options=["allset","moonlab", "qhgnn"]

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
                return jsonify(serialize(get_parameters()))

@api.route("/train/<model>")
class Train(Resource):
    @api.expect(train_params_model)
    def post(self, model: str):
        """Train a model with specified parameters (async)"""
        data = request.get_json() or {}

        if model not in options:
            return {"error": "Model not found."}, 404

        # Generate unique job ID
        job_id = f"{model}_{datetime.now().timestamp()}"

        # Start training in background thread
        thread = threading.Thread(target=train_model_async, args=(model, data, job_id))
        thread.daemon = True
        thread.start()

        return {
            "status": "accepted",
            "message": f"Training {model} model started in background",
            "job_id": job_id
        }, 202


def train_model_async(model: str, data: dict, job_id: str):
    """Run model training in a background thread"""
    training_jobs[job_id] = {
        "status": "running",
        "model": model,
        "started_at": datetime.now().isoformat(),
        "message": f"Training {model} model..."
    }

    try:
        match model:
            case "qhgnn":
                trainer = Train_QHGNN()
                trainer.train(
                    num_epochs=data.get("num_epochs", 1000),
                    lr=data.get("lr", 0.001),
                    hidden_layer_size=data.get("hidden_layer_size", 128),
                    train_proportion=data.get("train_proportion", 0.8),
                    dropout=data.get("dropout", 0.5),
                    weight_decay=data.get("weight_decay", 5e-4),
                    gamma=data.get("gamma", 0.5),
                    milestones_input=data.get("milestones_input", "50,100")
                )
            case "allset":
                pass
            case "moonlab":
                pass

    except Exception as e:
        print("Oops")
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
