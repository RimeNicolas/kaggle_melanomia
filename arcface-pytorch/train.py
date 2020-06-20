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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import Config
from data.dataset import Dataset
from models.focal_loss import *
from models.model_func import *
from test import TestModel

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class TrainModel:
    def __init__(self, opt=Config()):
        self.opt = opt
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_dataset(self):
        dataset = Dataset(self.opt.train_list, phase='train', input_shape=self.opt.input_shape)

        if self.opt.n_fold == 0:
            train_set = dataset
        else:
            self.split = int(0.90 * len(dataset))
            train_set, val_set = torch.utils.data.random_split(dataset, [self.split, len(dataset) - self.split])

        self.trainloader = torch.utils.data.DataLoader(train_set,
                                    batch_size=self.opt.train_batch_size,
                                    shuffle=True,
                                    num_workers=self.opt.num_workers)
        if self.opt.n_fold > 0:
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

        # if self.opt.epoch_start > 0:
        #     path_parameters = os.path.join(os.getcwd(), 'arcface-pytorch', 'checkpoints', self.opt.path_model_parameters)
        #     self.model.load_state_dict(torch.load(path_parameters))
        #     path_parameters = os.path.join(os.getcwd(), 'arcface-pytorch', 'checkpoints', self.opt.path_metric_parameters)
        #     self.metric_fc.load_state_dict(torch.load(path_parameters))
        self.model.to(self.device)
        if self.metric_fc is not None:
            self.metric_fc.to(self.device)
            if self.opt.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}, 
                                            {'params': self.metric_fc.parameters(), 'lr': self.opt.lr_metric}],
                                            lr=self.opt.lr_model, weight_decay=self.opt.weight_decay)
            else:
                self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, 
                                            {'params': self.metric_fc.parameters(), 'lr': self.opt.lr_metric}],
                                            lr=self.opt.lr_model, weight_decay=self.opt.weight_decay)
        else:
            if self.opt.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}],
                                            lr=self.opt.lr_model, weight_decay=self.opt.weight_decay)
            else:
                self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}],
                                            lr=self.opt.lr_model, weight_decay=self.opt.weight_decay)

        # self.scheduler = StepLR(self.optimizer, step_size=self.opt.lr_step*len(self.trainloader), gamma=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode = "min", patience = 5, verbose = True, factor = 0.2)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode = "max", patience = 3, verbose = True, factor = 0.2)

    def _output(self, data_input, label):
        data_input = data_input.to(self.device)
        feature = self.model(data_input)
        if self.metric_fc is not None:
            if self.opt.metric == 'linear':
                feature = self.metric_fc(feature)
            else:
                feature = self.metric_fc(feature, label)
        return feature

    def _opt_step(self, data):
        data_input, label = data
        label = label.to(self.device).long()
        output = self._output(data_input, label)
        loss = self.criterion(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

    def _eval_model(self, dataset_type='val'):
        if dataset_type == 'train':
            dataloader = self.trainloader
        elif dataset_type == 'val':
            dataloader = self.valloader
        else:
            Exception('no known dataset type to compute accuracy on')
        self.model.eval()
        with torch.no_grad():
            predictions, targets = list(), list()
            for ii, data in enumerate(dataloader):
                data_input, label = data
                if self.opt.metric == 'arc_margin':
                    label = label.to(self.device).long()
                output = self._output(data_input, label)
                predictions.append(torch.softmax(output, 1)[:, 1].detach().cpu().numpy())
                targets.append(label.detach().cpu().numpy())
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        auc = roc_auc_score(targets, predictions)
        acc = accuracy_score(targets, np.round(predictions))
        if dataset_type == 'train':
            print('accuracy on training set : {:.4f}, auc on training set : {:.4f}'.format(acc, auc))
        if dataset_type == 'val':
            self.val_auc = auc
            print('accuracy on validation set : {:.4f}, auc on validation set : {:.4f}'.format(acc, auc))
        self.model.train()

    def _info_training(self):
        print('#' * 100)
        print('start training: loss {}, opt {}, img_shape {}, w_decay {:.1e}, batch {}'.format(
                self.opt.loss, self.opt.optimizer, self.opt.input_shape[1], self.opt.weight_decay, 
                self.opt.train_batch_size))
        if self.opt.metric == 'linear':
            print('metric {}'.format(self.opt.metric))
        else:
            print('metric {}, s {}, margin {}'.format(self.opt.metric, self.opt.arc_s, self.opt.arc_m))
        print('#' * 100)


    def _train(self):
        self.init_training()
        self.t0 = time.time()
        self.t1 = self.t0
        self.val_auc = 0
        self.best_val_auc = 0
        # self._eval_model('train')
        for i in range(self.opt.epoch_max):
            self.model.train()

            self.running_loss = 0
            for _, data in enumerate(self.trainloader):
                self._opt_step(data)

            self.running_loss /= len(self.trainloader)

            time_str = time.asctime(time.localtime(time.time()))
            lr_s = list()
            for param_group in self.optimizer.param_groups:
                lr_s.append(param_group['lr'])
            print('{} train epoch {}, loss {:.5f}, time {:.2f} s, lr model {:.1e}, lr metric {:.1e}'.format(
                time_str, self.opt.epoch_start + i+1, self.running_loss, time.time() - self.t0, lr_s[0], lr_s[1]))

            if (i+1) % self.opt.accuracy_train_interval == 0:
                self._eval_model('train')
            if self.opt.n_fold > 0 :
                if (i+1) % self.opt.accuracy_val_interval == 0:
                    self._eval_model('val')

            # if self.val_auc > self.best_val_auc or (i+1) == self.opt.epoch_max:
            #     self.best_val_auc = self.val_auc
            #     if self.opt.save_model is True:
            #         save_model(self.model, self.opt.checkpoints_path, self.opt.backbone, self.opt.epoch_start + i+1)
            #         save_model(self.metric_fc, self.opt.checkpoints_path, 'metric', self.opt.epoch_start + i+1)
            if self.val_auc > self.best_val_auc:
                self.best_val_auc = self.val_auc
                if self.opt.save_model is True:
                    torch.save(self.model.state_dict(), os.path.join(self.opt.checkpoints_path, self.opt.backbone + '.pth'))
                    torch.save(self.metric_fc.state_dict(), os.path.join(self.opt.checkpoints_path, 'metric' + '.pth'))

            self.scheduler.step(self.running_loss)
            # self.scheduler.step(self.val_auc)

            self.t0 = time.time()
        print('total training time {:.2f} min'.format((time.time() - self.t1) / 60.0))

    def _train_fold(self):
        self._info_training()
        if self.opt.n_fold == 0:
            self._init_dataset()
            self._train()
        else:
            check_path = self.opt.checkpoints_path
            for i in range(self.opt.n_fold):
                self._init_dataset()
                self.opt.checkpoints_path = os.path.join(check_path, 'checkpoints' + str(i+1))
                os.makedirs(self.opt.checkpoints_path, exist_ok=True)
                self._train()
                print('=' * 100)
            self.opt.checkpoints_path = check_path

    def __call__(self):
        try:
            def handler(signal_received, frame):
                # Handle any cleanup here
                print('SIGINT or CTRL-C detected. Training stopped')
                exit(0)
            signal(SIGINT, handler)
            while True:
                self._train_fold()
                break
        except:
            print('error occured while training the model')
            raise


def tune_hyperparameters():
    np.random.seed(10)
    torch.manual_seed(10)
    s_s = [1, 3, 5]
    m_s = [0.0]
    training_model = TrainModel()
    for s in s_s:
        for m in m_s:
            training_model.opt.arc_m = m
            training_model.opt.arc_s = s
            print('-'*80)
            training_model()


def train_submission():
    cfg = Config()
    cfg.save_model = True
    # cfg.split_train_val = False
    TrainModel(cfg)()
    TestModel(cfg)()

                
if __name__ == '__main__':
    # TrainModel()()
    # tune_hyperparameters()
    train_submission()
