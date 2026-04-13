import time
from model.MoonLabHGNN.train import Train_MoonLabHGNN
from model.QualityHGNN.train import Train_QHGNN
from model.AllSetTransformer.train import Train_AllSetTransformer


if __name__ == "__main__":
    epochs = 10
    trainer = Train_QHGNN()
    start = time.time()
    trainer.train(num_epochs=epochs)
    elapsed = time.time() - start
    print(f"\n=== Total time for {epochs} epochs: {elapsed:.2f}s ===")
    #trainer = Train_MoonLabHGNN()
    #trainer.train()
    #trainer = Train_AllSetTransformer()
    #trainer.train()
