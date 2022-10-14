import numpy as np

import config as config
import metrics as metrics
import gaussian_utils

from trainer import MultiSourceTrainer

from tqdm import tqdm

import torch
from torch.autograd import Variable

def get_individual_performance():
    # Computes the target performance for source-only and post-adaptation models
    # First outputs source only performance for each source domain, followed by target performance for each source domain
    pre_adapt_accs = []
    post_adapt_accs = []
    
    # First, see the initial performance of the models
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        # trainer_S.load_model_weights()
        trainer_S.load_model_weights(it_thresh='enough_iter')

        pre_adapt_accs.append(trainer_S.val_over_target_set(save_weights=False))
        
    # Next, the performance after adaptation
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        # trainer_S.load_model_weights('model_max_iter' + str(trainer_S.settings['max_iter']) + '.pth')
        trainer_S.load_model_weights(it_thresh='max_iter')

        post_adapt_accs.append(trainer_S.val_over_target_set(save_weights=False))
    
    return np.asarray(pre_adapt_accs), np.asarray(post_adapt_accs)

def get_target_accuracy(weights, it_thresh='max_iter'):
    # Computes the accuracy obtained by combining logits from several models

    logit_sum = None

    all_logits_dict = {}
    all_labels_dict = {}

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        trainer_S.load_model_weights(it_thresh=it_thresh)
        
        trainer_S.set_mode(trainer_S.settings['mode']['val'])

        with torch.no_grad():
            # Gather samples from both source domains
            all_labels_tar                  =  []
            all_preds_tar                   =  []
            all_F_src                       =  []
            all_logits                      =  []

            dom = trainer_S.trgt_domain

            trainer_S.initialize_target_val_dataloader()

            for i in range(trainer_S.val_target_dataset.img.shape[0]):
                images = trainer_S.val_target_dataset.img[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
                label  = trainer_S.val_target_dataset.label[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
                if images.shape[0] == 0:
                    continue
                
                x                           = images.to(trainer_S.settings['device']).float()
                label                       = label.to(trainer_S.settings['device']).long()
                F                           = trainer_S.network.model['Fs'](x)
                C                           = trainer_S.network.model['C'](F)

                cls_logits,_,mat            = metrics.get_logits(feats={'C':C}, num_cls_heads=trainer_S.settings['num_cls_heads'])
                all_logits.append(cls_logits)

                all_labels_tar.extend(list(label.cpu().numpy()))

            all_labels_tar = np.asarray(all_labels_tar)
            all_logits = torch.cat(all_logits, dim=0).cpu().numpy()

            all_logits_dict[src_domain_idx] = np.copy(all_logits)
            all_labels_dict[src_domain_idx] = np.copy(all_labels_tar)

            if logit_sum is None:
                logit_sum = weights[src_domain_idx] * all_logits
            else:
                logit_sum += weights[src_domain_idx] * all_logits
    
    labels_hat = np.argmax(logit_sum, axis=-1)
    
    return np.mean(all_labels_tar == labels_hat), labels_hat, all_labels_tar

def get_combined_predictions(weights, it_thresh='max_iter'):
    # Computes the accuracy obtained by combining logits from several models

    logit_sum = None

    all_logits_dict = {}
    all_labels_dict = {}

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        trainer_S.load_model_weights(it_thresh=it_thresh)
        
        trainer_S.set_mode(trainer_S.settings['mode']['val'])

        with torch.no_grad():
            # Gather samples from both source domains
            all_labels_tar                  =  []
            all_preds_tar                   =  []
            all_F_src                       =  []
            all_logits                      =  []

            dom = trainer_S.trgt_domain

            trainer_S.initialize_target_val_dataloader()

            for i in range(trainer_S.val_target_dataset.img.shape[0]):
                images = trainer_S.val_target_dataset.img[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
                label  = trainer_S.val_target_dataset.label[i*trainer_S.batch_size : (i+1) * trainer_S.batch_size]
                if images.shape[0] == 0:
                    continue
                
                x                           = images.to(trainer_S.settings['device']).float()
                label                       = label.to(trainer_S.settings['device']).long()
                F                           = trainer_S.network.model['Fs'](x)
                C                           = trainer_S.network.model['C'](F)

                cls_logits,_,mat            = metrics.get_logits(feats={'C':C}, num_cls_heads=trainer_S.settings['num_cls_heads'])
                all_logits.append(cls_logits)

                all_labels_tar.extend(list(label.cpu().numpy()))

            all_labels_tar = np.asarray(all_labels_tar)
            all_logits = torch.cat(all_logits, dim=0).cpu().numpy()

            all_logits_dict[src_domain_idx] = np.copy(all_logits)
            all_labels_dict[src_domain_idx] = np.copy(all_labels_tar)

            if logit_sum is None:
                logit_sum = weights[src_domain_idx] * all_logits
            else:
                logit_sum += weights[src_domain_idx] * all_logits
    
    labels_hat = np.argmax(logit_sum, axis=-1)

    confidence_counts = {}
    print(np.max(logit_sum, axis=1))
    for conf in range(100):
        c = conf/100
        idx = np.max(logit_sum, axis=1) > c
        print("for confidence {} there are {} samples and target accuracy is {}".format(c, np.sum(idx), np.mean(labels_hat[idx] == all_labels_tar[idx])))
        confidence_counts[c] = np.sum(idx)
    
    return confidence_counts, trainer_S.val_target_dataset.img.shape[0], labels_hat, all_labels_tar


def learn_w_w2(it_thresh='max_iter', num_steps=100):
    w2_dist = np.zeros(len(config.settings['src_datasets']))

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        dom = config.settings['src_datasets'][src_domain_idx]
        trainer = MultiSourceTrainer(src_domain_idx)
        
        # learn the gaussians
        trainer.load_model_weights(it_thresh='enough_iter')
        gaussian_utils.learn_gaussians(trainer)
        n_samples = np.ones(trainer.N_CLASSES, dtype=int) * trainer.settings["gaussian_samples_per_class"]
        trainer.gaussian_z, trainer.gaussian_y = gaussian_utils.sample_from_gaussians(trainer.means, trainer.covs, n_samples)

        # Load source and target dataloaders
        trainer.initialize_src_train_dataloader()
        trainer.initialize_target_adapt_dataloader()

        # Find the batch size to use
        batch_max = min(trainer.source_dataset_train.img.shape[0], trainer.adapt_target_dataset_train.img.shape[0])
        batch_max = min(batch_max, trainer.adapt_batch_size)
        trainer.batch_size = trainer.adapt_batch_size = batch_max

        # Load model weights for source-only and post adaptation
        src_trainer = MultiSourceTrainer(src_domain_idx)
        src_trainer.load_model_weights(it_thresh='enough_iter')
        src_trainer.set_mode(src_trainer.settings['mode']['val'])

        adapt_trainer = MultiSourceTrainer(src_domain_idx)
        adapt_trainer.load_model_weights(it_thresh='max_iter')
        adapt_trainer.set_mode(adapt_trainer.settings['mode']['val'])
        
        # Get the mean W2 distance between encodings of the current domain and the target domain
        w2_dist[src_domain_idx] = 0
        for step in range(num_steps):
            # Get source samples
            X_src,_ = trainer.source_dataset_train.sample(trainer.batch_size)
            X_src = Variable(X_src).to(trainer.settings['device']).float()

            X_tar,_ = trainer.adapt_target_dataset_train.sample(trainer.adapt_batch_size)
            X_tar = Variable(X_tar).to(trainer.settings['device']).float()
                
            # Compute the number of gaussian samples to be used for the current batch
            normalized_dist = np.ones(trainer.N_CLASSES, dtype=int) / trainer.N_CLASSES
            num_samples = np.array(normalized_dist * trainer.adapt_batch_size, dtype=int)
            while batch_max > np.sum(num_samples):
                idx = np.random.choice(range(trainer.N_CLASSES), p = normalized_dist)
                num_samples[idx] += 1

            # Get gaussian samples for the current batch
            gz = []
            gy = []
            for c in range(trainer.N_CLASSES):
                ind = np.where(trainer.gaussian_y == c)[0]
                ind = ind[np.random.choice(range(len(ind)), num_samples[c], replace=False)]
                gz.append(trainer.gaussian_z[ind])
                gy.append(trainer.gaussian_y[ind])
            gz = np.vstack(gz)
            gy = np.concatenate(gy)
            gz = torch.as_tensor(gz).to(trainer.settings['device']).float()
            gy = torch.as_tensor(gy).to(trainer.settings['device']).long()

            # Compute predictions
            with torch.no_grad():
                f_src = src_trainer.network.model['Fs'](X_src)
                f_tar = adapt_trainer.network.model['Fs'](X_tar)
                
                d1 = metrics.sliced_wasserstein_distance(f_src, gz, trainer.settings['num_projections'], 2, trainer.settings['device']).item()
                d2 = metrics.sliced_wasserstein_distance(f_tar, gz, trainer.settings['num_projections'], 2, trainer.settings['device']).item()
                w2_dist[src_domain_idx] += d1 + d2

            if step % (num_steps // 10) == 0:
                print(step, w2_dist / (step + 1), dom)

    w2_dist = w2_dist / num_steps
    print("Summation w = {}".format(w2_dist))
        
    w = 1 / w2_dist
    w = w / np.sum(w)

    print("Final w = {}".format(w))
        
    return w


# Learn the weights w minimizing the generalizability objective
# Note: Not source free, as we require simultaneous access to multiple source domains
def learn_w_generalizability(it_thresh='max_iter', num_steps=300):   
    trainers = {}
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        dom = config.settings['src_datasets'][src_domain_idx]
        trainers[dom] = MultiSourceTrainer(src_domain_idx)
        trainers[dom].load_model_weights(it_thresh=it_thresh)
        trainers[dom].set_mode(trainers[dom].settings['mode']['val'])

        trainers[dom].src_data = {}
        trainers[dom].initialize_src_train_dataloader()

    w_gen = np.zeros(len(config.settings['src_datasets']))

    for d1_idx in range(len(config.settings['src_datasets'])):
        for d2_idx in range(len(config.settings['src_datasets'])):
            d1 = config.settings['src_datasets'][d1_idx]
            d2 = config.settings['src_datasets'][d2_idx]

            if d1 == d2:
                continue

            acc = 0
            for i in range(trainers[d2].source_dataset_train.img.shape[0]):
                images = trainers[d2].source_dataset_train.img[i*trainers[d2].batch_size : (i+1) * trainers[d2].batch_size]
                label  = trainers[d2].source_dataset_train.label[i*trainers[d2].batch_size : (i+1) * trainers[d2].batch_size]
                if images.shape[0] == 0:
                    continue

                x                           = images.to(trainers[d2].settings['device']).float()
                label                       = label.to(trainers[d2].settings['device']).long()
            
                with torch.no_grad():
                    Fs = trainers[d1].network.model['Fs'](x)
                    C  = trainers[d1].network.model['C'](Fs)
            
                cls_logits,_,_          = metrics.get_logits(feats={'C':C}, num_cls_heads=trainers[d1].settings['num_cls_heads'])
                cls_confs,cls_preds     = torch.max(cls_logits,dim=-1)

                acc += torch.sum(cls_preds == label).item()
            acc = acc / trainers[d2].source_dataset_train.img.shape[0]

            w_gen[d1_idx] += acc

    w_gen = w_gen / np.sum(w_gen)

    print("Generalizability weights = {}".format(w_gen))
        
    return w_gen


'''
    Compare the methods by how many high confidence samples they have
'''
def learn_w_high_confidence(it_thresh='enough_iter', confidence=.5):
    w_hc = np.zeros(len(config.settings['src_datasets']))

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        trainer = MultiSourceTrainer(src_domain_idx)
        trainer.load_model_weights(it_thresh=it_thresh)
        trainer.set_mode(trainer.settings['mode']['val'])
        trainer.initialize_target_val_dataloader()

        for i in range(trainer.val_target_dataset.img.shape[0]):
            images = trainer.val_target_dataset.img[i*trainer.batch_size : (i+1) * trainer.batch_size]
            label  = trainer.val_target_dataset.label[i*trainer.batch_size : (i+1) * trainer.batch_size]
            if images.shape[0] == 0:
                continue

            x                           = images.to(trainer.settings['device']).float()
            label                       = label.to(trainer.settings['device']).long()
        
            with torch.no_grad():
                Fs = trainer.network.model['Fs'](x)
                C  = trainer.network.model['C'](Fs)
        
            cls_logits,_,_          = metrics.get_logits(feats={'C':C}, num_cls_heads=trainer.settings['num_cls_heads'])
            cls_confs,cls_preds     = torch.max(cls_logits,dim=-1)

            w_hc[src_domain_idx] += torch.sum(cls_confs > confidence).item()
        
    print("Number of high confidence samples for each domain = {}".format(w_hc))

    w_hc_raw = np.copy(w_hc)
    w_hc = w_hc / np.sum(w_hc)

    print("High confidence w = {}".format(w_hc))
        
    return w_hc, w_hc_raw, trainer.val_target_dataset.img.shape[0]

#### Each dataset has their own representation of the source domains!!!
def test_predictions(it_thresh='max_iter'):
    trainers = {}
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        trainers[src_domain_idx] = MultiSourceTrainer(src_domain_idx)
        trainers[src_domain_idx].load_model_weights(it_thresh=it_thresh)
        trainers[src_domain_idx].set_mode(trainers[src_domain_idx].settings['mode']['val'])
        trainers[src_domain_idx].initialize_target_val_dataloader()

    with torch.no_grad():
        all_labels_trgt = []
        all_Fs = {}
        all_Cs = {}
        all_logits = {}
        all_confs = {}
        all_preds = {}
        all_preds_trgt                   =  {}

        for src_domain_idx in range(len(config.settings['src_datasets'])):
            all_Fs[src_domain_idx] = []
            all_Cs[src_domain_idx] = []
            all_confs[src_domain_idx] = []
            all_preds[src_domain_idx] = []
            all_preds_trgt[src_domain_idx] = []
            all_logits[src_domain_idx] = []

        for src_domain_idx in range(len(config.settings['src_datasets'])):
            for i in range(trainers[0].val_target_dataset.img.shape[0]):
                images = trainers[src_domain_idx].val_target_dataset.img[i*trainers[0].batch_size : (i+1) * trainers[0].batch_size]
                label  = trainers[src_domain_idx].val_target_dataset.label[i*trainers[0].batch_size : (i+1) * trainers[0].batch_size]
                if images.shape[0] == 0:
                    continue

                x                           = images.to(trainers[0].settings['device']).float()
                label                       = label.to(trainers[0].settings['device']).long()
            
                Fs = trainers[src_domain_idx].network.model['Fs'](x)
                C  = trainers[src_domain_idx].network.model['C'](Fs)

                all_Fs[src_domain_idx].extend(Fs.clone().detach().cpu().numpy())
                all_Cs[src_domain_idx].extend(C.clone().detach().cpu().numpy())
            
                cls_logits,_,_          = metrics.get_logits(feats={'C':C}, num_cls_heads=trainers[src_domain_idx].settings['num_cls_heads'])
                cls_confs,cls_preds     = torch.max(cls_logits,dim=-1)

                all_logits[src_domain_idx].extend(cls_logits.clone().cpu().numpy())
                all_confs[src_domain_idx].extend(cls_confs.clone().cpu().numpy())
                all_preds[src_domain_idx].extend(cls_preds.clone().cpu().numpy())

                all_preds_trgt[src_domain_idx].extend(list(cls_preds.cpu().numpy()))

                if src_domain_idx == 0:
                    all_labels_trgt.extend(list(label.cpu().numpy()))

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        all_Fs[src_domain_idx] = np.vstack(all_Fs[src_domain_idx])
        all_Cs[src_domain_idx] = np.vstack(all_Cs[src_domain_idx])
        all_logits[src_domain_idx] = np.vstack(all_logits[src_domain_idx])
        all_preds[src_domain_idx] = np.asarray(all_preds[src_domain_idx])
        all_labels_trgt = np.asarray(all_labels_trgt)

    # Accuracies for each domain
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("{} -> {}, adaptation target acc = {:.4f}".format(trainers[src_domain_idx].src_domain, trainers[src_domain_idx].trgt_domain, np.sum(all_preds[src_domain_idx] == all_labels_trgt) / len(all_labels_trgt)))