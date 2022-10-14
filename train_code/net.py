
import torch.nn as nn
import torch
from customlayers import ClassifierLayer as C
from customlayers import BackBoneLayer as G
from customlayers import ForwardLayer as F
import config as config

class SingleSourceNet(nn.Module):
    
    def __init__(self, settings): 
        
        super(SingleSourceNet, self).__init__()

        self.model = {}
        to_train = config.settings['to_train']
        
        for module in to_train:
            self.model[module]={}

            dropout = settings['dropout'][settings['dataset_name']]

            if module =='Fs':
                self.model[module] = F(config.settings['bb_output'],config.settings['bb_output']//2,config.settings['F_dims'], dropout)
            elif module =='C':
                self.model[module] = C(config.settings['F_dims'], config.settings['num_C'][config.settings['src_datasets'][0]] * config.settings['num_cls_heads'], dropout)
            elif module =='G':
                self.model[module] = G(config.settings['bb'],config.settings['bb_output'])
            elif module in config.settings['ss_tasks']:
                self.model[module] = C(config.settings['F_dims'], 4)

        for module,compts in self.model.items():
            self.add_module(module,compts)


    def forward(self, x ):
        raise NotImplementedError('Implemented a custom forward in train loop')

if __name__=='__main__':
    raise NotImplementedError('Please check README.md for execution details')
