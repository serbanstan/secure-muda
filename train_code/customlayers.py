import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torchvision import models


# single domain classifier layer
class ClassifierLayer(nn.Module):
    def __init__(self, input_dim, classes, dropout):
        super(ClassifierLayer, self).__init__()
        self.classes = classes
        self.input_dim = input_dim

        self.net = nn.Sequential(
                nn.Dropout(dropout),
                weightNorm(nn.Linear(self.input_dim, self.classes))
            )

    def forward(self, x):
        return self.net(x)

# forward layer
class ForwardLayer(nn.Module):
 
    def __init__(self, inp_lin1, inp_lin2, f_dims, dropout):
        super(ForwardLayer, self).__init__()
        self.inp_lin1 = inp_lin1
        self.inp_lin2 = inp_lin2
        self.f_dims = f_dims
        
        self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.inp_lin1,self.inp_lin2),
                nn.BatchNorm1d(self.inp_lin2),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.inp_lin2,self.inp_lin2),
                nn.BatchNorm1d(self.inp_lin2),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.inp_lin2,self.f_dims),
                nn.BatchNorm1d(self.f_dims),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.f_dims, self.f_dims),
                nn.BatchNorm1d(self.f_dims),
                nn.ELU(inplace=True)
            )
            
    def forward(self,x):
        return self.net(x)


# backbone layer
class BackBoneLayer(nn.Module):

    def __init__(self,pre,out_feats):
        super(BackBoneLayer, self).__init__()

        if pre == 'resnet101':
            temp_resnet = models.resnet101(pretrained=True)
            self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])
        elif pre == 'resnet50':
            temp_resnet = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])
        
        self.pre = pre
        self.out_feats = out_feats

    def forward(self, x):
        feats = self.features(x)
        return feats.view((x.shape[0], self.out_feats))

