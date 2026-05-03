import time
import optuna
from model.QualityHGNN_V2.train import Train_QHGNN_v2
from model.MoonLabHGNN.train import Train_MoonLabHGNN


def objective(trial, trainer):
    lr                = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden_layer_size = trial.suggest_categorical("hidden_layer_size", [64, 128, 256, 512])
    dropout           = trial.suggest_float("dropout", 0.1, 0.7)
    weight_decay      = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    quality_weight    = trial.suggest_float("quality_weight", 0.0, 2.0)  # QHGNN only
    patience          = trial.suggest_int("patience", 50, 200)  # QHGNN only
    gamma             = trial.suggest_float("gamma", 0.1, 0.9)

    result = trainer.train(
        num_epochs=2000,
        lr=lr,
        hidden_layer_size=hidden_layer_size,
        dropout=dropout,
        weight_decay=weight_decay,
        quality_weight=quality_weight,  # QHGNN only
        patience=patience,  # QHGNN only
        gamma=gamma,
        seed=42
    )
    return result.valid_acc  # percentage (0–100), higher = better


if __name__ == "__main__":
    N_TRIALS = 30

    start = time.time()

    trainer = Train_QHGNN_v2()  # DB and preprocessing happen exactly once here
    #trainer = Train_MoonLabHGNN()  # DB and preprocessing happen exactly once here

    study = optuna.create_study(
        direction="maximize",
        study_name="Moonlab-accuracy",
        storage="sqlite:///moonlab_study.db",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, trainer), n_trials=N_TRIALS)
    
    elapsed = time.time() - start

    best = study.best_trial
    print(f"\n=== Optuna search complete ({N_TRIALS} trials) in {elapsed:.2f}s ===")
    print(f"Best trial #{best.number}  accuracy={best.value:.2f}%")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
