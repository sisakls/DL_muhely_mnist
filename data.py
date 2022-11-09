import torch
import torchvision
import torchvision.transforms as tfs
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import datasets
#https://github.com/huggingface/datasets/issues/4112 - atyauristen


def task_split(data, label_list):
    task_dict = {}
    
    for ind, lbl_set in enumerate(label_list):
        task = data.filter(lambda x: x['label'] in lbl_set)
        task_dict['task_{}'.format(ind)] = task
        
    return datasets.DatasetDict(task_dict)


def transform_func(examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = tfs.ToTensor()
    
    examples["image"] = [transforms(img) for img in examples["image"]] #transforms(img).to(device)
    
    return examples


def collate_fun(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["image"]))
        labels.append(example["label"])
        
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return (images, labels)


def prep_data():
    data = datasets.load_dataset("mnist")
    
    label_list = [{0,1}, {2,3}, {4,5}, {6,7}, {8,9}]
    tasks = task_split(data['train'], label_list)
    #print(tasks)

    ds_train = tasks['task_0'].with_transform(transform_func) #egyelore csak task_0
    ds_test  = data['test'].with_transform(transform_func)
    
    #ds_train = ds_train.with_format('torch')
    #ds_test = ds_test.with_format('torch')
    
    #print(ds_train, ds_test)
    train_loader = torch.utils.data.DataLoader(ds_train, collate_fn=collate_fun, batch_size=32)
    test_loader  = torch.utils.data.DataLoader(ds_test, collate_fn=collate_fun, batch_size=32)
    
    #print(iter(train_loader).next()[0].shape)

    return train_loader, test_loader


#def prep_data():
#    data = datasets.load_dataset("mnist")
#    
#    label_list = [{0,1}, {2,3}, {4,5}, {6,7}, {8,9}]
#    tasks = task_split(data['train'], label_list)
#    
#    ds_train = tasks.with_transform(transform_func)
#    ds_test  = data['test'].with_transform(transform_func)
#    
#    ds_train = ds_train.with_format('torch')
#    ds_test = ds_test.with_format('torch')
#    
#    train_loader = torch.utils.data.DataLoader(ds_train['task_0'], batch_size=32) #egyelore csak task_0
#    test_loader  = torch.utils.data.DataLoader(ds_test, batch_size=32)
#    
#    return train_loader, test_loader