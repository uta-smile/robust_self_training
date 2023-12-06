
import torch.nn as nn
import torch.nn.functional as F


class ResidueEmbedding(nn.Embedding):
    def __init__(self, vocab_size=21, embed_size=128, padding_idx=None):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)

        
class GRUnet(nn.Module):
    def __init__(self,lstm_hdim=1024, embed_size=128, num_layers=3, bidirectional=True, lstm=False, outsize=4, dropout=0.0):
        super().__init__()
        """
            This version of the model has all the bells & whistles (e.g. 
            dropconnect) ripped out so its slimmed down for inference
            
        """

        self.lstm_hdim = lstm_hdim
        self.embed = ResidueEmbedding(vocab_size=22, embed_size=embed_size, padding_idx=21)
        self.lstm = nn.GRU(128, 1024, num_layers=3, bidirectional=True, batch_first=True, dropout=dropout)
        self.outlayer = nn.Linear(lstm_hdim*2, outsize)
        self.finalact = F.log_softmax

    def forward(self, x):
        """
            Assumes a batch size of one currently but can be changed
        """
        x = self.embed(x) # torch.Size([8, 5980, 128])
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.outlayer(x)
        x = self.finalact(x, dim=-1)
        x = x.permute(0, 2, 1)
        # print('x3', x.shape)
        return x.squeeze()        
        
        
class S4PRED(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        """
            This loads the ensemble of models in a lazy way but its clear and 
            leaves the weight loading out of the run_model script. 
        """

        # Manually listing for clarity and hot swapping in future
        self.model_1 = GRUnet(dropout=dropout)
        # self.model_2=GRUnet()
        # self.model_3=GRUnet()
        # self.model_4=GRUnet()
        # self.model_5=GRUnet()
        
    def forward(self, x):
        y_1 = self.model_1(x)

        # input('debug')
        # y_2=self.model_2(x)
        # print(2)
        # y_3=self.model_3(x)
        # print(3)
        # y_4=self.model_4(x)
        # print(4)
        # y_5=self.model_5(x)
        # print(fine-tuning-0)
        # y_out=y_1*0.2+y_2*0.2+y_3*0.2+y_4*0.2+y_5*0.2
        return y_1
        
        
