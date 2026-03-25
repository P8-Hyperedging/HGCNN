from model.MoonLabHGNN.train import Train_MoonLabHGNN
from model.QualityHGNN.train import Train_QHGNN
from model.AllSetTransformer.train import Train_AllSetTransformer


if __name__ == "__main__":
    trainer = Train_QHGNN()
    trainer.train()
    trainer = Train_MoonLabHGNN()
    trainer.train()
    # trainer = Train_AllSetTransformer()
    #trainer.train()
