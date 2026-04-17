import time
from model.MoonLabHGNN.train import Train_MoonLabHGNN
from model.QualityHGNN.train import Train_QHGNN
from model.AllSetTransformer.train import Train_AllSetTransformer
from model.QualityHGNN_V2.train import Train_QHGNN_v2


if __name__ == "__main__":
    epochs = 100
    start = time.time()    

    trainer = Train_QHGNN_v2()
    trainer.train(num_epochs=epochs)
    
    #trainer = Train_MoonLabHGNN()
    #trainer.train()
    #trainer = Train_AllSetTransformer()
    #trainer.train()

    elapsed = time.time() - start
    print(f"\n=== Total time for {epochs} epochs: {elapsed:.2f}s ===")

