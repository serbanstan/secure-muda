from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from PIL import Image

# This file works for RGB images.

def load_data(data_folder, domain_name, batch_size, phase='train', train_val_split=True, train_ratio=.8):
    transform_dict = {
        'train': transforms.Compose(
            # [transforms.Resize(256),
            [transforms.Resize((256,256)),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'test': transforms.Compose(
            # [transforms.Resize(224),
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}

    data = datasets.ImageFolder(root='/'.join([data_folder, domain_name]), transform=transform_dict[phase])

    if 'domain-net' not in data_folder:
        if phase == 'train':
            if train_val_split:
                train_size = int(train_ratio * len(data))
                test_size = len(data) - train_size
                data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
                train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True,
                                                        num_workers=4)
                val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False,
                                                    num_workers=4)
                return [train_loader, val_loader]
            else:
                train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                        num_workers=4)
                return train_loader
        else: 
            test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,
                                                        num_workers=4)
            return test_loader
    else:
        print('Domain-net specific processing')

        data_imgs_idx = {k[0]:v for v,k in enumerate(data.imgs)}

        train_idx_f = '/'.join([data_folder, domain_name]) + '_train.txt'
        test_idx_f = '/'.join([data_folder, domain_name]) + '_test.txt'

        with open(train_idx_f) as f:
            train_imgs = f.readlines()
        train_imgs = ['/'.join([data_folder, x.split(" ")[0]]) for x in train_imgs]
        train_indices = [data_imgs_idx[x] for x in train_imgs]

        with open(test_idx_f) as f:
            test_imgs = f.readlines()
        test_imgs = ['/'.join([data_folder, x.split(" ")[0]]) for x in test_imgs]
        test_indices = [data_imgs_idx[x] for x in test_imgs]

        train_loader = DataLoader(Subset(data, train_indices), batch_size=batch_size, shuffle=True, drop_last=True,
                                                        num_workers=4)
        test_loader = DataLoader(Subset(data, test_indices), batch_size=batch_size, shuffle=False, drop_last=False,
                                                        num_workers=4)

        print('Domain {} has {} samples. Train dataloader has size {} and test dataloader has size {}'.format(domain_name, len(data), \
            len(train_loader.dataset), \
            len(test_loader.dataset)))

        return train_loader, test_loader


# ## Below are for ImageCLEF datasets

# class ImageCLEF(torch.utils.data.Dataset):
#     def __init__(self, root_dir, domain, transform=None):
#         super(ImageCLEF, self).__init__()
#         self.transform = transform
#         file_name = root_dir + 'list/' + domain + 'List.txt'
#         lines = open(file_name, 'r').readlines()
#         self.images, self.labels = [], []
#         self.domain = domain
#         for item in lines:
#             line = item.strip().split(' ')
#             self.images.append(root_dir + domain + '/' + line[0].split('/')[-1])
#             self.labels.append(int(line[1].strip()))

#     def __getitem__(self, index):
#         image = self.images[index]
#         target = self.labels[index]
#         img = Image.open(image).convert('RGB')
#         if self.transform:
#             image = self.transform(img)
#         return image, target

#     def __len__(self):
#         return len(self.images)

# def load_imageclef_train(root_path, domain, batch_size, phase):
#     transform_dict = {
#         'src': transforms.Compose(
#             [transforms.Resize((256, 256)),
#              transforms.RandomCrop(224),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ]),
#         'tar': transforms.Compose(
#             [transforms.Resize((224, 224)),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ])}
#     data = ImageCLEF(root_dir=root_path, domain=domain, transform=transform_dict[phase])
#     train_size = int(0.8 * len(data))
#     test_size = len(data) - train_size
#     data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
#     train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False,
#                                                num_workers=4)
#     val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=False,
#                                              num_workers=4)
#     return train_loader, val_loader

# def load_imageclef_test(root_path, domain, batch_size, phase):
#     transform_dict = {
#         'src': transforms.Compose(
#             [transforms.Resize((256,256)),
#              transforms.RandomCrop(224),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ]),
#         'tar': transforms.Compose(
#             [transforms.Resize((224, 224)),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ])}
#     data = ImageCLEF(root_dir=root_path, domain=domain, transform=transform_dict[phase])
#     data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
#     return data_loader