import os
import numpy as np
import random
import time
from signal import signal, SIGINT
from sys import exit

import torch
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from config.config import Config
from data.dataset import Dataset
from models.focal_loss import *
from models.model_func import *
from test import TestModel

class TrainModel:
    def __init__(self, opt=Config()):
        self.opt = opt
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_dataset()

    def init_dataset(self):
        if self.opt.dataset_type == 'random':
            dataset = RandomDataset(128, 1024)
        else:
            dataset = Dataset(self.opt.train_list, phase='train', input_shape=self.opt.input_shape)

        self.split_train_val = self.opt.split_train_val
        if self.split_train_val:
            self.split = int(0.90 * len(dataset))
            train_set, val_set = torch.utils.data.random_split(dataset, [self.split, len(dataset) - self.split])
        else:
            train_set = dataset

        self.trainloader = torch.utils.data.DataLoader(train_set,
                                    batch_size=self.opt.train_batch_size,
                                    shuffle=True,
                                    num_workers=self.opt.num_workers)
        if self.split_train_val:
            self.valloader = torch.utils.data.DataLoader(val_set,
                                    batch_size=self.opt.train_batch_size,
                                    shuffle=False,
                                    num_workers=self.opt.num_workers)
    
    def init_training(self):
        if self.opt.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma=2)
        else:
            # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.45, 0.55]).to(self.device))
            self.criterion = torch.nn.CrossEntropyLoss()

        self.model = init_model(self.opt)
        self.metric_fc = init_metric(self.opt)

        # print(model)
        if self.opt.epoch_start > 0:
            path_parameters = os.path.join(os.getcwd(), 'arcface-pytorch', 'checkpoints', self.opt.path_model_parameters)
            self.model.load_state_dict(torch.load(path_parameters))
            path_parameters = os.path.join(os.getcwd(), 'arcface-pytorch', 'checkpoints', self.opt.path_metric_parameters)
            self.metric_fc.load_state_dict(torch.load(path_parameters))
        self.model.to(self.device)
        if self.metric_fc is not None:
            self.metric_fc.to(self.device)
            if self.opt.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                                            lr=self.opt.lr, weight_decay=self.opt.weight_decay)
            else:
                self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.metric_fc.parameters(), 'lr': 10*self.opt.lr}],
                                            lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        else:
            if self.opt.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}],
                                            lr=self.opt.lr, weight_decay=self.opt.weight_decay)
            else:
                self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}],
                                            lr=self.opt.lr, weight_decay=self.opt.weight_decay)

        self.scheduler = StepLR(self.optimizer, step_size=self.opt.lr_step*len(self.trainloader), gamma=0.1)

    def _output(self, data_input, label):
        data_input = data_input.to(self.device)
        feature = self.model(data_input)
        if self.metric_fc is not None:
            if self.opt.metric == 'linear':
                feature = self.metric_fc(feature)
            else:
                feature = self.metric_fc(feature, label)
        return feature

    def opt_step(self, data):
        data_input, label = data
        label = label.to(self.device).long()
        output = self._output(data_input, label)
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.running_loss += loss.item()

    def _eval_model(self, dataset_type='val'):
        self.model.eval()
        if dataset_type == 'train':
            dataloader = self.trainloader
        elif dataset_type == 'val':
            dataloader = self.valloader
        else:
            Exception('no known dataset type to compute accuracy on')
        acc = 0
        for ii, data in enumerate(dataloader):
            data_input, label = data
            if self.opt.metric == 'arc_margin':
                label = label.to(self.device).long()
            output = self._output(data_input, label)
            output = torch.argmax(output, axis=1).data.cpu()
            label = label.data.cpu()
            aa = torch.sum(output.long() == label.long())
            acc += aa.item()
        acc /= len(dataloader.dataset)
        if dataset_type == 'train':
            print('accuracy on training set : {:.4f}'.format(acc))
        if dataset_type == 'val':
            print('accuracy on validation set : {:.4f}'.format(acc))
        self.model.train()


    def _train(self):
        self.init_training()
        self.t0 = time.time()
        self.t1 = self.t0
        print('start training: loss {}, opt {}, img_shape {}, w_decay {:.1e}, batch {}'.format(
                self.opt.loss, self.opt.optimizer, self.opt.input_shape[1], self.opt.weight_decay, self.opt.train_batch_size))
        # self._eval_model('train')
        for i in range(self.opt.epoch_max):
            self.model.train()

            self.running_loss = 0
            for ii, data in enumerate(self.trainloader):
                self.opt_step(data)

            self.running_loss /= len(self.trainloader)
            time_str = time.asctime(time.localtime(time.time()))
            lr_s = list()
            for param_group in self.optimizer.param_groups:
                lr_s.append(param_group['lr'])
            print('{} train epoch {}, loss {:.5f}, time {:.2f} s, lr model {:.1e}, lr metric {:.1e}'.format(
                time_str, self.opt.epoch_start + i+1, self.running_loss, time.time() - self.t0, lr_s[0], lr_s[1]))

            if (i+1) % self.opt.save_interval == 0 or (i+1) == self.opt.epoch_max:
                if self.opt.save_model is True:
                    save_model(self.model, self.opt.checkpoints_path, self.opt.backbone, self.opt.epoch_start + i+1)
                    save_model(self.metric_fc, self.opt.checkpoints_path, 'metric', self.opt.epoch_start + i+1)
            if (i+1) % self.opt.accuracy_train_interval == 0:
                self._eval_model('train')
            if self.split_train_val is True:
                if (i+1) % self.opt.accuracy_val_interval == 0:
                    self._eval_model('val')

            self.t0 = time.time()
        print('total training time {:.2f} min'.format((time.time() - self.t1) / 60.0))

    def __call__(self):
        try:
            def handler(signal_received, frame):
                # Handle any cleanup here
                print('SIGINT or CTRL-C detected. Training stopped')
                exit(0)
            signal(SIGINT, handler)
            while True:
                self._train()
                break
        except:
            print('error occured while training the model')
            raise


def tune_hyperparameters():
    list_weight_decay = [5, 1]
    training_model = TrainModel()
    for w_decay in list_weight_decay:
        training_model.opt.weight_decay = w_decay
        print('-'*50)
        training_model()


def train_submission():
    cfg = Config()
    cfg.save_model = True
    cfg.split_train_val = False
    # TrainModel(cfg)()
    TestModel(cfg)()

                
if __name__ == '__main__':
    # TrainModel()()
    # tune_hyperparameters()
    train_submission()
