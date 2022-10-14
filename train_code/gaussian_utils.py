import numpy as np

import metrics as metrics

from tqdm import tqdm

import torch
import time

from sklearn.mixture import GaussianMixture

def sample_from_gaussians(means, covs, n_samples, perm_res=True):
    # Return samples from the num_classes gaussians trained on the source 
    N_CLASSES = len(n_samples)

    Xs = []
    Ys = []
    
    for i in range(N_CLASSES):
        if n_samples[i] > 0:
            # print(i, means[i], covs[i])
            curr_x = np.random.multivariate_normal(means[i], covs[i], n_samples[i])
            curr_y = np.repeat(i, n_samples[i])
                        
            Xs.append(curr_x)
            Ys.append(curr_y)

    Xs = np.vstack(Xs)
    Ys = np.concatenate(Ys)

    if not perm_res:
        return Xs, Ys
    else:
        perm = np.random.permutation(Xs.shape[0])
        return Xs[perm,:], Ys[perm]

def learn_gaussians(trainer_S, debug=False):
    ###########################################################################################################################
    # Extract source domain latent features from all sources
    ###########################################################################################################################s
    trainer_S.set_mode(trainer_S.settings['mode']['val'])

    with torch.no_grad():
        # Gather samples from both source domains
        all_labels_src                  =  []
        all_preds_src                   =  []
        all_confs_src                   =  []
        all_F_src                       =  []
        all_C_src                       =  []

        dom = trainer_S.src_domain
            
        for i in range(trainer_S.source_dataset_train.img.shape[0]):
            images = trainer_S.source_dataset_train.img[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
            label  = trainer_S.source_dataset_train.label[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
            if images.shape[0] == 0:
                continue

            x                           = images.to(trainer_S.settings['device']).float()
            label                       = label.to(trainer_S.settings['device']).long()
            # G                           = trainer_S.network.model['G'](x)
            F                           = trainer_S.network.model['Fs'](x)
            C                           = trainer_S.network.model['C'](F)

            cls_logits,_,mat            = metrics.get_logits(feats={'C':C}, num_cls_heads=trainer_S.settings['num_cls_heads'])
            cls_confs,cls_preds         = torch.max(cls_logits,dim=-1)

            all_labels_src.extend(list(label.cpu().numpy()))
            all_preds_src.extend(list(cls_preds.cpu().numpy()))
            all_confs_src.extend(list(cls_confs.cpu().numpy()))
            all_F_src.append(F)
            all_C_src.append(C)

        all_labels_src = np.asarray(all_labels_src)
        all_preds_src = np.asarray(all_preds_src)
        all_confs_src = np.asarray(all_confs_src)
        all_F_src = torch.cat(all_F_src,dim=0).cpu().numpy()
        all_C_src = torch.cat(all_C_src,dim=0).cpu().numpy()

    ###########################################################################################################################
    # Learn means and covariances
    ###########################################################################################################################
    adapt_lvl = all_F_src

    N_CLASSES = trainer_S.settings['num_C'][trainer_S.src_domain]
    Z_SIZE = adapt_lvl.shape[-1]

    trainer_S.means = np.zeros((N_CLASSES, Z_SIZE))
    trainer_S.covs = np.zeros((N_CLASSES, Z_SIZE, Z_SIZE))

    for c in range(N_CLASSES):
        idx = (all_labels_src == c) & (all_preds_src == c)

        if np.sum(idx) == 0:
            idx = all_labels_src == c

        assert np.sum(idx) > 0

        trainer_S.means[c] = np.mean(adapt_lvl[idx], axis=0)
        trainer_S.covs[c] = np.dot((adapt_lvl[idx] - trainer_S.means[c]).T, (adapt_lvl[idx] - trainer_S.means[c])) / np.sum(idx)
        

    if debug == True:
        # Store intermediate results for debug purposes
        trainer_S.adapt_lvl = adapt_lvl
        trainer_S.all_labels_src = all_labels_src
