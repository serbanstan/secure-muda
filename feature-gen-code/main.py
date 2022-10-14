"""
Extract features from pre-trained networks.
The main procedures are finetune and extract features.
Finetune: Given an Imagenet pretrained model (such as ResNet50), finetune it on a dataset (we call it source)
Extractor: After fine-tune, extract features on the target domain using finetuned models on source

This class supports most image models: Alexnet, Resnet(xx), VGG.
Other text or digit models can be easily extended using this code, see models.py for details.
"""

import argparse
import data_load
import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import copy
import os

# Command setting
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--model_name', type=str,
                    help='model name', default='resnet50')
parser.add_argument('--batchsize', type=int, help='batch size', default=64)
parser.add_argument('--gpu', type=int, help='cuda id', default=0)
parser.add_argument('--dataset', type=str, default='office-31')
parser.add_argument('--source', type=str, default='amazon')
# parser.add_argument('--target', type=str, default='webcam')
parser.add_argument('--target', type=str, default='webcam',
                    help='List of target domains separated by commas')
parser.add_argument('--num_class', type=int, default=12)
parser.add_argument('--dataset_path', type=str,
                    default='../../data/office-31/')
parser.add_argument('--epoch', type=int, help='Train epochs', default=100)
parser.add_argument('--momentum', type=float, help='Momentum', default=.9)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--finetune', type=int,
                    help='Needs finetune or not', default=1)
parser.add_argument('--extract', type=int,
                    help='Needs extract features or not', default=1)
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = {'src': int(args.batchsize), 'tar': int(args.batchsize)}


def get_optimizer(model):
    learning_rate = args.lr
    param_group = []
    param_group += [{'params': model.base_network.parameters(),
                     'lr': learning_rate}]
    param_group += [{'params': model.classifier_layer.parameters(),
                     'lr': learning_rate * 10}]
    optimizer = optim.SGD(param_group, momentum=args.momentum)
    return optimizer


# Schedule learning rate according to DANN if you want to (while I think this equation is wierd therefore I did not use this one)
def lr_schedule(optimizer, epoch):
    def lr_decay(LR, n_epoch, e):
        return LR / (1 + 10 * e / n_epoch) ** 0.75

    for i in range(len(optimizer.param_groups)):
        if i < len(optimizer.param_groups) - 1:
            optimizer.param_groups[i]['lr'] = lr_decay(
                LEARNING_RATE, N_EPOCH, epoch)
        else:
            optimizer.param_groups[i]['lr'] = lr_decay(
                LEARNING_RATE, N_EPOCH, epoch) * 10


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.to(DEVICE)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

def finetune(model, dataloaders, optimizer, criterion, best_model_path, use_lr_schedule=False):
    N_EPOCH = args.epoch
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()
    best_acc = 0.0
    acc_hist = []

    for epoch in range(1, N_EPOCH + 1):
        if use_lr_schedule:
            lr_schedule(optimizer, epoch)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders[phase].dataset)
            acc_hist.append([epoch_loss, epoch_acc])
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
                                                                                  epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(
                ), 'save_model_{}/best_{}_{}-{}.pth'.format(args.dataset, args.model_name, args.source, epoch))
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_pass // 60, time_pass % 60))
    print('------Best acc: {}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), best_model_path)
    print('Best model saved!')
    return model, best_acc, acc_hist


# Extract features for given intermediate layers
# Currently, this only works for ResNet since AlexNet and VGGNET only have features and classifiers modules.
# You will need to manually define a function in the forward function to extract features
# (by letting it return features and labels).
# Please follow digit_deep_network.py for reference.
class FeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model._modules['module'] if type(
            model) == torch.nn.DataParallel else model
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.model._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def extract_feature(model, dataloader, save_path, load_from_disk=True, model_path=''):
    if load_from_disk:
        model = models.Network(base_net=args.model_name,
                               n_class=args.num_class)
        model.load_state_dict(torch.load(model_path))
        model = model.to(DEVICE)
    model.eval()
    correct = 0
    fea_all = torch.zeros(1,1+model.base_network.output_num()).to(DEVICE)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            feas = model.get_features(inputs)
            labels = labels.view(labels.size(0), 1).float()
            x = torch.cat((feas, labels), dim=1)
            fea_all = torch.cat((fea_all, x), dim=0)
            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1]
            # correct += torch.sum(preds == labels.data.long())
            correct += torch.sum(preds == labels.view(labels.size(0)).data.long())
            # print(inputs.shape, preds.shape, outputs.shape, labels.shape, torch.sum(preds == labels.view(labels.size(0)).data.long()))
        # print(correct, len(dataloader.dataset))
        test_acc = correct.double() / len(dataloader.dataset)
    fea_numpy = fea_all.cpu().numpy()
    np.savetxt(save_path, fea_numpy[1:], fmt='%.6f', delimiter=',')
    print('Test acc: %f' % test_acc)

# You may want to classify with 1nn after getting features


def classify_1nn(data_train, data_test):
    '''
    Classification using 1NN
    Inputs: data_train, data_test: train and test csv file path
    Outputs: yprediction and accuracy
    '''
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    data = {'src': np.loadtxt(data_train, delimiter=','),
            'tar': np.loadtxt(data_test, delimiter=','),
            }
    Xs, Ys, Xt, Yt = data['src'][:, :-1], data['src'][:, -
                                                      1], data['tar'][:, :-1], data['tar'][:, -1]
    Xs = StandardScaler(with_mean=0, with_std=1).fit_transform(Xs)
    Xt = StandardScaler(with_mean=0, with_std=1).fit_transform(Xt)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, Ys)
    ypred = clf.predict(Xt)
    acc = accuracy_score(y_true=Yt, y_pred=ypred)
    print('Acc: {:.4f}'.format(acc))
    return ypred, acc


if __name__ == '__main__':
    torch.manual_seed(10)

    # Load data
    print('Loading data...')
    data_folder = args.dataset_path
    domain = {'src': str(args.source), 'tar': str(args.target)}
    dataloaders = {}
    data_train = data_load.load_data(
        data_folder, domain['src'], BATCH_SIZE['src'], 'train', train_val_split=True, train_ratio=.8)
    dataloaders['train'], dataloaders['val'] = data_train[0], data_train[1]
    print('Data loaded: Source: {}, Target: {}'.format(args.source, args.target))

    # Finetune
    if args.finetune == 1:
        print('Begin finetuning...')
        net = models.Network(base_net=args.model_name,
                             n_class=args.num_class).to(DEVICE)
        # criterion = nn.CrossEntropyLoss()
        criterion = CrossEntropyLabelSmooth(num_classes=args.num_class, epsilon=0.1) 
        optimizer = get_optimizer(net)
        if not os.path.exists('save_model_{}/'.format(args.dataset)):
            os.mkdir('save_model_{}/'.format(args.dataset))
        save_path = 'save_model_{}/best_{}_{}.pth'.format(
            args.dataset, args.model_name, args.source)
        model_best, best_acc, acc_hist = finetune(
            net, dataloaders, optimizer, criterion, save_path, use_lr_schedule=False)
        print('Finetune completed!')

    # Extract features from finetuned model
    if args.extract == 1:
        model_path = 'save_model_{}/best_{}_{}.pth'.format(
            args.dataset, args.model_name, args.source)

        for target_domain in args.target.split(','):
            data_test = data_load.load_data(data_folder, target_domain, BATCH_SIZE['tar'], 'test')
            dataloaders['adapt'], dataloaders['test'] = data_train[0], data_train[1]

            if 'domain-net' not in data_folder:
                feature_save_path = 'save_model_{}/{}_{}_{}.csv'.format(args.dataset, args.source, target_domain, args.model_name)
                print(feature_save_path)
                extract_feature(None, data_test, feature_save_path, load_from_disk=True, model_path=model_path)
            else:
                train_feature_save_path = 'save_model_{}/{}_{}_{}_train.csv'.format(args.dataset, args.source, target_domain, args.model_name) 
                test_feature_save_path = 'save_model_{}/{}_{}_{}_test.csv'.format(args.dataset, args.source, target_domain, args.model_name)

                extract_feature(None, data_test[0], train_feature_save_path, load_from_disk=True, model_path=model_path)
                extract_feature(None, data_test[1], test_feature_save_path, load_from_disk=True, model_path=model_path)

            print('Deep features are extracted and saved!')
