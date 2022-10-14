import numpy as np
import copy
import shutil
import os

import config as config
import metrics as metrics
from net import SingleSourceNet as smodel
from dataset import FeatureDataset, FrozenDataset
import gaussian_utils

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.utils.weight_norm as weightNorm

import matplotlib.pyplot as plt

import time

class MultiSourceTrainer(object):
    def __init__(self, src_domain_idx):
        self.settings                                   = copy.deepcopy(config.settings)
        
        self.src_domain                                 = self.settings['src_datasets'][src_domain_idx]
        self.trgt_domain                                = self.settings['trgt_datasets'][0]
        self.N_CLASSES                                  = self.settings["num_C"][self.src_domain]

        self.network                                    = smodel(self.settings).to(self.settings['device'])

        self.mixing_weights                             = None

        self.to_train                                   = self.settings['to_train']

        self.pseudo_target_dist                         = None

        # Batch size
        self.batch_size                                 = self.settings['batch_size']
        self.val_batch_size                             = self.settings['val_batch_size_factor']*self.settings['batch_size']
        self.adapt_batch_size                           = self.settings['adapt_batch_size']
        
        self.current_iteration                          = self.settings['start_iter']
        self.exp_name                                   = self.settings['exp_name']
        self.phase                                      = self.settings['mode']['train']

        # Datasets
        self.source_dataset_train                       = None
        self.target_dataset_val                         = None
        self.val_target_dataset                         = None
        self.adapt_target_dataset_train                 = None

        self.itt_delete                                 = []
        self.best_src_val_acc                           = -1
        
        self.initialize_src_train_dataloader()
        # self.initialize_src_train_self_sup_dataloader()
        self.init_optimizers()
        self.disable_dropout() # dropout initially disabled

        all_losses                                      = self.optimizer_dict.keys()
        self.active_losses                              = [current_loss for current_loss in all_losses if  self.settings['use_loss'][current_loss]]
        
        self.source_loss_history                        = []
        self.adaptation_loss_history                    = {}
        self.target_acc_history                         = []
        self.it_history                                 = []
        
        assert self.settings['enough_iter'] % self.settings['val_after'] == 0
        assert self.settings['max_iter'] % self.settings['val_after'] == 0
        
    def init_folder_paths(self):
        for name in ['weights_path', 'summaries_path']:
            if not os.path.exists(self.settings[name]):
                os.mkdir(self.settings[name])
            if not os.path.exists(os.path.join(self.settings[name],self.settings['exp_name'])):
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name']))
            if not os.path.exists(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain)):
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
            else:
                shutil.rmtree(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
    
    '''
    Function to load model weights
    '''
    def load_model_weights(self, it_thresh='enough_iter', weights_file=None):
        if weights_file == None:
            weights_file = 'model_' + it_thresh + str(self.settings[it_thresh]) + '.pth'
        load_weights_path = os.path.join(self.settings['weights_path'],self.settings['exp_name'], self.src_domain, weights_file)

        print("Loading path = {}".format(load_weights_path))
        
        dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
        model_state_dict = dict_to_load['model_state_dict']

        for module,compts in self.network.model.items():
            self.network.model[module].load_state_dict(model_state_dict[module])

    '''
    Function to load optimizer
    '''
    def load_optimizers(self, opt_file=None):
        if opt_file == None:
            opt_file = 'opt_enough_iter' + str(self.settings['enough_iter']) + '.pth'
        load_weights_path = os.path.join(self.settings['weights_path'], self.settings['exp_name'], self.src_domain, opt_file)
        dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
        optimizer_state_dict = dict_to_load['optimizer_state_dict']

        for name,optimizer in self.optimizer_dict.items():
            if self.settings['use_loss'][name]:
                optimizer.load_state_dict(optimizer_state_dict[name])
    
    def check_and_save_weights(self):
        if self.current_iteration <= self.settings['enough_iter']:
            val_acc = self.val_over_source_set()

            if self.best_src_val_acc <= val_acc:
                self.best_src_val_acc = val_acc
                self.save_weights()
                print("Saving at iteration {} with src. val accuracy = {}".format(self.current_iteration, self.best_src_val_acc))
        else:
            if self.current_iteration == self.settings['max_iter']:
                self.save_weights()
            
    '''
    Function to save model and optimizer state
    '''
    def save_weights(self):
        bkp_iter = self.current_iteration
        if self.current_iteration <= self.settings['enough_iter']:
            self.current_iteration = self.settings['enough_iter']

        weights_path = self.settings['weights_path']

        model_state_dict={}
        for module,compts in self.network.model.items():
            model_state_dict[module]=compts.cpu().state_dict()

        optimizer_state_dict ={}
        for name,optimizer in self.optimizer_dict.items():
            optimizer_state_dict[name]=optimizer.state_dict()
            
            
        save_dict    = {
                         'model_state_dict':model_state_dict,
                       }
        save_path = os.path.join(self.settings['weights_path'], self.exp_name, self.src_domain)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for it_thresh in ['enough_iter', 'max_iter']:
            if self.current_iteration == self.settings[it_thresh]:
                torch.save(save_dict, os.path.join(save_path, 'model_' + it_thresh + str(self.current_iteration) + '.pth'))
            
        torch.save(save_dict, os.path.join(save_path, 'model_' + str(self.current_iteration) + '.pth'))
        self.network.to(self.settings['device'])

        
        save_dict    = {
                         'optimizer_state_dict':optimizer_state_dict,
                       }
        
        for it_thresh in ['enough_iter', 'max_iter']:
            if self.current_iteration == self.settings[it_thresh]:
                torch.save(save_dict, os.path.join(save_path, 'opt_' + it_thresh + str(self.current_iteration) + '.pth'))
            
        torch.save(save_dict, os.path.join(save_path, 'opt_' + str(self.current_iteration) + '.pth'))
        self.network.to(self.settings['device'])

        self.current_iteration = bkp_iter

    def save_summaries(self):
        save_path = os.path.join(self.settings['summaries_path'], self.exp_name, self.src_domain)

        np.savetxt(os.path.join(save_path, "exp_details" ), [self.settings['comments']], fmt="%s")
        np.savetxt(os.path.join(save_path, "source_loss"), self.source_loss_history, fmt="%.5f")
        for loss_type in self.adaptation_loss_history:
            np.savetxt(os.path.join(save_path, "adaptation_loss_{}".format(loss_type)), self.adaptation_loss_history[loss_type], fmt="%.5f")
        np.savetxt(os.path.join(save_path, "target_accuracy"), self.target_acc_history, fmt="%.5f")
        np.savetxt(os.path.join(save_path, "distribution"), self.pseudo_target_dist, fmt="%.5f")

        plt.title('Source log loss for domain {}'.format(self.src_domain))
        plt.plot(np.log(self.source_loss_history))
        plt.savefig(os.path.join(save_path, "source_loss_plot"))
        plt.clf()
        
        for loss_type in self.adaptation_loss_history:
            plt.title('Adaptation log {} loss for domain {} -> {}'.format(loss_type, self.src_domain, self.trgt_domain))
            plt.plot(np.log(self.adaptation_loss_history[loss_type]))
            plt.savefig(os.path.join(save_path, "adaptation_{}_loss_plot".format(loss_type)))
            plt.clf()

        plt.title('Target accuracy')
        plt.plot(self.it_history, self.target_acc_history)
        plt.savefig(os.path.join(save_path, "target_accuracy_plot"))
        plt.clf()

    def load_summaries(self):
        save_path = os.path.join(self.settings['summaries_path'], self.exp_name, self.src_domain)

        self.source_loss_history = np.loadtxt(os.path.join(save_path, "source_loss"))
        self.adaptation_loss_history = np.loadtxt(os.path.join(save_path, "adaptation_loss_total"))
        self.target_acc_history = np.loadtxt(os.path.join(save_path, "target_accuracy"))
        
    '''
    Utility Functions to initialize source and target dataloaders
    '''
    def initialize_src_train_dataloader(self):
        assert 0 < self.settings['train_ratio'][self.settings['dataset_name']] and self.settings['train_ratio'][self.settings['dataset_name']] <= 1

        if self.source_dataset_train == None:
            if 'domain-net' not in self.settings['dataset_name']:
                self.source_dataset_train            = FrozenDataset("{}_{}.csv".format(self.src_domain, self.src_domain), \
                    self.settings['train_ratio'][self.settings['dataset_name']])

                # for x in self.settings['src_datasets']:
                #     if x != self.src_domain:
                #         self.source_dataset_other[x] = FrozenDataset("{}_{}.csv".format(self.src_domain, x), self.settings['train_ratio'][self.settings['dataset_name']])
            else:
                self.source_dataset_train            = FrozenDataset("{}_{}_train.csv".format(self.src_domain, self.src_domain), 1.0)

                source_val_data                      = FrozenDataset("{}_{}_test.csv".format(self.src_domain, self.src_domain), 1.0)
                self.source_dataset_train.val_img    = source_val_data.img
                self.source_dataset_train.val_label  = source_val_data.label

    def initialize_target_val_dataloader(self):
        if self.val_target_dataset == None:
            if 'domain-net' not in self.settings['dataset_name']:
                self.val_target_dataset = FrozenDataset("{}_{}.csv".format(self.src_domain, self.trgt_domain))
            else:
                self.val_target_dataset = FrozenDataset("{}_{}_test.csv".format(self.src_domain, self.trgt_domain))

    def initialize_target_adapt_dataloader(self):
        if self.adapt_target_dataset_train == None:
            if 'domain-net' not in self.settings['dataset_name']:
                self.adapt_target_dataset_train = FrozenDataset("{}_{}.csv".format(self.src_domain, self.trgt_domain))
            else:
                self.adapt_target_dataset_train = FrozenDataset("{}_{}_train.csv".format(self.src_domain, self.trgt_domain))
    
    '''
    Utility function to set the model in eval or train mode
    '''
    def set_mode(self,mode):
        self.phase = mode

        if self.phase == self.settings['mode']['train']:
            self.network.train()
        elif self.phase == self.settings['mode']['val']:
            self.network.eval()
        
    '''
    Initializing optimizers
    '''
    def init_optimizers(self):
        self.optimizer_dict  = {}
        # self.scheduler_dict  = {}
        to_train = self.settings['to_train']
        for loss_name,loss_details in self.settings['optimizer_dict'].items():
            if self.settings['use_loss'][loss_name]:
                opt_param_list = []
                for comp in loss_details:
                    if comp in to_train:
                        opt_param_list.append({'params':self.network.model[comp].parameters(), 'lr':self.settings['lr'][loss_name], 'weight_decay':5e-4})
                self.optimizer_dict[loss_name] = optim.Adam(params = opt_param_list)

    '''
    Target dataset validation
    '''
    def val_over_target_set(self, save_weights=True):
        self.set_mode(self.settings['mode']['val'])
        self.initialize_target_val_dataloader()

        with torch.no_grad():
            all_labels_trgt                  =  []
            all_preds_trgt                   =  []

            for i in range(self.val_target_dataset.img.shape[0]):
                images = self.val_target_dataset.img[i*self.batch_size : (i+1) * self.batch_size]
                label  = self.val_target_dataset.label[i*self.batch_size : (i+1) * self.batch_size]
                if images.shape[0] == 0:
                    continue

                x                           = images.to(self.settings['device']).float()
                label                       = label.to(self.settings['device']).long()

                F                           = self.network.model['Fs'](x)
                C                           = self.network.model['C'](F)
                
                cls_logits,_,mat            = metrics.get_logits(feats={'C':C}, num_cls_heads=self.settings['num_cls_heads'])
                cls_confs,cls_preds         = torch.max(cls_logits,dim=-1)

                all_labels_trgt.extend(list(label.cpu().numpy()))
                all_preds_trgt.extend(list(cls_preds.cpu().numpy()))
                
            if save_weights:
                self.target_acc_history.append(metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt}))
                self.it_history.append(self.current_iteration)
            else:
                print("target accuracy at iteration {} = {}".format(self.current_iteration, metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt})))
            

            return metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt})

    '''
    Source dataset validation
    '''
    def val_over_source_set(self, save_weights=True):
        self.set_mode(self.settings['mode']['val'])
        # self.initialize_src_val_dataloader()
        self.initialize_src_train_dataloader()

        with torch.no_grad():
            all_labels_src                  =  []
            all_preds_src                   =  []
            
            for i in range(self.source_dataset_train.val_img.shape[0]):
                images = self.source_dataset_train.val_img[i*self.batch_size : (i+1) * self.batch_size]
                label  = self.source_dataset_train.val_label[i*self.batch_size : (i+1) * self.batch_size]
                if images.shape[0] == 0:
                    continue

                x                           = images.to(self.settings['device']).float()
                label                       = label.to(self.settings['device']).long()

                F                           = self.network.model['Fs'](x)
                C                           = self.network.model['C'](F)

                cls_logits,_,mat            = metrics.get_logits(feats={'C':C}, num_cls_heads=self.settings['num_cls_heads'])
                cls_confs,cls_preds         = torch.max(cls_logits,dim=-1)                

                all_labels_src.extend(list(label.cpu().numpy()))
                all_preds_src.extend(list(cls_preds.cpu().numpy()))

            return metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_src,'cls_preds':all_preds_src})
                
    '''
    Function to calculate the loss value
    '''
    def get_loss(self,which_loss):
        assert which_loss in ['source', 'target']

        if which_loss == 'source':
            src_C                   = self.src_features['C']
            src_labels              = self.src_features['labels']

            ce_loss                 = metrics.CrossEntropyLabelSmooth(num_classes=self.N_CLASSES, epsilon=0.1)(src_C, src_labels)

            loss = ce_loss

            self.source_loss_history.append(loss.item())
        elif which_loss == 'target':
            trgt_F                  = self.trgt_features['F']
            trgt_C                  = self.trgt_features['C']

            ####################################################################################################
            # Conditional Entropy loss
            ####################################################################################################
            start_time = time.time()

            logits,_,_  = metrics.get_logits(feats={'C':trgt_C}, num_cls_heads=self.settings['num_cls_heads'])
            if self.settings['dataset_name'] in ['office-home', 'domain-net']:
                softmax_ = nn.Softmax(dim=1)(logits) # slow down convergence
            else:
                softmax_ = logits

            entropy = metrics.Entropy(softmax_) 
            conditional_entropy_loss = torch.mean(entropy)
            conditional_entropy_loss = self.settings['gamma'][self.settings['dataset_name']] * conditional_entropy_loss

            if self.current_iteration % self.settings['val_after']== 0 and self.settings['verbose']==True:
                print("Computed entropy loss in {:.4f}".format(time.time() - start_time))

            if 'conditional_entropy' in self.adaptation_loss_history:
                self.adaptation_loss_history['conditional_entropy'].append(conditional_entropy_loss.item())
            else:
                self.adaptation_loss_history['conditional_entropy'] = [conditional_entropy_loss.item()]


            ####################################################################################################
            # W2 loss
            ####################################################################################################
            start_time = time.time()

            # Compute the number of gaussian samples to be used for the current batch
            normalized_dist = self.pseudo_target_dist / np.sum(self.pseudo_target_dist)
            num_samples = np.array(normalized_dist * self.adapt_batch_size, dtype=int)
            while self.adapt_batch_size > np.sum(num_samples):
                idx = np.random.choice(range(self.N_CLASSES), p = normalized_dist)
                num_samples[idx] += 1

            # Get gaussian samples for the current batch
            gz = []
            gy = []
            for c in range(self.N_CLASSES):
                ind = torch.where(self.gaussian_y == c)[0]
                perm = torch.randperm(ind.shape[0])
                ind = ind[perm][:num_samples[c]]
                gz.append(self.gaussian_z[ind])
                gy.append(self.gaussian_y[ind])
            gz = torch.cat(gz)
            gy = torch.cat(gy)

            if self.current_iteration % self.settings['val_after']== 0 and self.settings['verbose']==True:
                print("Sampling from Gaussians took {:.4f}".format(time.time() - start_time))

            w2_loss = metrics.sliced_wasserstein_distance(trgt_F, gz, self.settings['num_projections'], 2, self.settings['device'])

            if 'w2' in self.adaptation_loss_history:
                self.adaptation_loss_history['w2'].append(w2_loss.item())
            else:
                self.adaptation_loss_history['w2'] = [w2_loss.item()]

            if self.current_iteration % self.settings['val_after']== 0 and self.settings['verbose']==True:
                print("Computed W2 loss in {:.4f}".format(time.time() - start_time))

            loss = w2_loss + conditional_entropy_loss

            if 'total' in self.adaptation_loss_history:
                self.adaptation_loss_history['total'].append(loss.item())
            else:
                self.adaptation_loss_history['total'] = [loss.item()]
        
        return loss
    
    '''
    Function to select active losses
    '''
    def loss(self):
        optim           = self.optimizer_dict[self.active_losses[self.current_loss]]
        optim.zero_grad()
        loss            = self.get_loss(self.active_losses[self.current_loss])
        loss.backward()
        optim.step()
                
    '''
    Function to implement the forward prop for a single source
    '''
    def forward(self):
        self.set_mode(self.settings['mode']['train'])

        if self.active_losses[self.current_loss] == 'source':
            # Computing the values for the source domain
            self.src_features                               = {}
            
            images,labels                                   = self.src_data['images'],self.src_data['labels']
            feats_F                                         = self.network.model['Fs'](images)
            feats_C                                         = self.network.model['C'](feats_F)

            self.src_features['F']              = feats_F
            self.src_features['C']              = feats_C
            self.src_features['labels']         = labels

        elif self.active_losses[self.current_loss] == 'target':
            start_time = time.time()

            # Computing target domain info
            self.trgt_features                              = {}

            images,labels                                   = self.trgt_data['images'],self.trgt_data['labels']

            # During adaptation, keep the feature extractor frozen
            feats_F                                         = self.network.model['Fs'](images)
            feats_C                                         = self.network.model['C'](feats_F)

            self.trgt_features['F']              =  feats_F
            self.trgt_features['C']              =  feats_C
            self.trgt_features['labels']         =  labels

            if self.current_iteration % self.settings['val_after']== 0 and self.settings['verbose']==True:
                print("Forward pass of the network in {:.4f}".format(time.time() - start_time))

    def disable_dropout(self):
        for module in self.network.model['Fs'].net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        for module in self.network.model['C'].net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def enable_dropout(self):
        for module in self.network.model['Fs'].net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        for module in self.network.model['C'].net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    '''
    Function for training the the data
    This function is called at every iteration
    '''
    def train(self):
        self.src_data                           = {}
        self.trgt_data                          = {}

        self.current_loss = 0
        if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
            self.current_loss = 1

        cond_1 = self.active_losses[self.current_loss] not in self.settings['losses_after_enough_iters']
        cond_2 = self.current_iteration <= max(self.settings['val_after'],self.settings['enough_iter'])

        # self.initialize_target_adapt_dataloader()
        
        if (cond_1 and cond_2) or (not cond_2):
            if self.current_iteration <= max(self.settings['val_after'],self.settings['enough_iter']):
                self.src_data['images'], self.src_data['labels'] = self.source_dataset_train.sample(self.batch_size)
                self.src_data['images'] = Variable(self.src_data['images']).to(self.settings['device']).float()
                self.src_data['labels'] = Variable(self.src_data['labels']).to(self.settings['device']).long()
            else:
                # Distribution matching between target domain and gaussians
                if self.current_iteration == max(self.settings['val_after'], self.settings['enough_iter']) + 1:
                    # Set the learning rate appropriately
                    self.init_optimizers()
                    self.disable_dropout()
                    
                    print()
                    print("STARTING ADAPTION FOR MODEL TRAINED ON {}".format(self.src_domain))

                    self.pseudo_target_dist = np.ones(self.N_CLASSES, dtype=int)

                    print("FINALIZED DISTRIBUTION FOR ADAPTATION ON {}:".format(self.trgt_domain))
                    print(self.pseudo_target_dist)
                    
                    # Learn Gaussians
                    gaussian_utils.learn_gaussians(self)

                    n_samples = np.ones(self.N_CLASSES, dtype=int) * self.settings["gaussian_samples_per_class"]
                    self.gaussian_z, self.gaussian_y = gaussian_utils.sample_from_gaussians(self.means, self.covs, n_samples)
                    self.gaussian_z = torch.as_tensor(self.gaussian_z).to(self.settings['device']).float()
                    self.gaussian_y = torch.as_tensor(self.gaussian_y).to(self.settings['device']).long()

                    self.initialize_target_adapt_dataloader()

                    # Keep the classifiers frozen
                    for param in self.network.model['C'].parameters():
                        param.requires_grad = False

                start_time = time.time()
                self.trgt_data['images'],self.trgt_data['labels'] = self.adapt_target_dataset_train.sample(self.adapt_batch_size)
                self.trgt_data['images'] = Variable(self.trgt_data['images']).to(self.settings['device']).float()
                self.trgt_data['labels'] = Variable(self.trgt_data['labels']).to(self.settings['device']).long()
                if self.current_iteration % self.settings['val_after']== 0 and self.settings['verbose']==True:
                    print("Sampled batch in {:.4f}".format(time.time() - start_time))

            # Correctly set networks and optimizers for source training or adaptation
            if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
                self.disable_dropout()
            else:
                self.enable_dropout()
            
            self.forward()
            self.loss()