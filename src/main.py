import time
import csv
from datetime import datetime
from model.QualityHGNN_V2.train import Train_QHGNN_v2
from model.MoonLabHGNN.train import Train_MoonLabHGNN


def run_qhgnn_ablation(seed=30000):
    """Run QHGNN ablation study with corruption levels 0-100 in steps of 10"""
    
    OUTPUT_CSV = f"qhgnn_ablation_results_seed{seed}.csv"
    
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["corruption_pct", "test_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for corruption_pct in range(0, 101, 10):
        print(f"Corruption: {corruption_pct}%")

        
        try:
            trainer = Train_QHGNN_v2(corruption_percentage=corruption_pct)
            result = trainer.train(
                num_epochs=500,
                lr=0.00074,
                hidden_layer_size=128,
                train_proportion=0.8,
                dropout=0.16,
                weight_decay=0.000021,
                gamma=0.62,
                seed=seed,
                patience=2000,
                quality_weight=0.76
            )
            
            row = {
                "corruption_pct": corruption_pct,
                "test_acc": result.test_acc,
            }
            
            with open(OUTPUT_CSV, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
            
            print(f"Test: {result.test_acc:.2f}%")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
            row = {
                "corruption_pct": corruption_pct,
                "test_acc": "ERROR",
            }
            with open(OUTPUT_CSV, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
    
    print(f"Done: {OUTPUT_CSV}")


def run_moonlab_ablation(seed=30000):
    """Run MoonLab ablation study with corruption levels 0-100 in steps of 10"""
    
    OUTPUT_CSV = f"moonlab_ablation_results_seed{seed}.csv"
    
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["corruption_pct", "test_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for corruption_pct in range(0, 101, 10):
        print(f"Corruption: {corruption_pct}%")
        
        try:
            trainer = Train_MoonLabHGNN()
            result = trainer.train(
                num_epochs=500,
                lr=0.00074,
                hidden_layer_size=128,
                train_proportion=0.8,
                dropout=0.16,
                weight_decay=0.000021,
                gamma=0.62,
                seed=seed,
                corruption_percentage=corruption_pct
            )
            
            row = {
                "corruption_pct": corruption_pct,
                "test_acc": result.test_acc,
            }
            
            with open(OUTPUT_CSV, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
            
            print(f"Test: {result.test_acc:.2f}%")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
            row = {
                "corruption_pct": corruption_pct,
                "test_acc": "ERROR",
            }
            with open(OUTPUT_CSV, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
    
    print(f"Done: {OUTPUT_CSV}")


if __name__ == "__main__":
    # Uncomment which one to run:
    #run_qhgnn_ablation()
    run_moonlab_ablation()
