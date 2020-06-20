import os
import time
import numpy as np

from collections import Counter

from config.config import Config
from data.dataset import Dataset
from models.focal_loss import *
from models.model_func import *


class TestModel:
    def __init__(self, opt=Config()):
        self.opt = opt
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def _init_dataset(self):
        test_set = Dataset(self.opt.test_list, phase='test', input_shape=self.opt.input_shape)

        self.testloader = torch.utils.data.DataLoader(test_set,
                                    batch_size=self.opt.test_batch_size,
                                    shuffle=False,
                                    num_workers=self.opt.num_workers)

    def _init_model(self):
        self.model = init_model(self.opt)
        path_parameters = os.path.join(self.opt.checkpoints_path, self.opt.backbone + '.pth')
        self.model.load_state_dict(torch.load(path_parameters))
        self.model.to(self.device)
        self.model.eval()

        self.metric_fc = init_metric(self.opt)
        path_parameters = os.path.join(self.opt.checkpoints_path, 'metric.pth')
        self.metric_fc.load_state_dict(torch.load(path_parameters))
        self.metric_fc.to(self.device)
        self.metric_fc.eval()

    def _output(self, data_input):
        data_input = data_input.to(self.device)
        feature = self.model(data_input)
        output = self.metric_fc(feature)
        return output
        # return torch.softmax(output, 1)[:,1]

    # def _infer_old(self):
    #     self.t0 = time.time()
    #     cnt = Counter()
    #     print('start testing')
    #     for ii, data in enumerate(self.testloader):
    #         # if ii > 1:
    #         #     break
    #         if (ii+1) % 100 == 0:
    #             print('batch {} of {}'.format(ii+1, len(self.testloader)))
    #         data_input, img_name = data
    #         output = self._output(data_input)
    #         output = list(output.detach().cpu().numpy())
    #         cnt += Counter(np.round(output))
    #         self.csv_data += zip(img_name, output)
    #     self.csv_data = sorted(self.csv_data, key = lambda x : x[0].split('.')[0])
    #     print('inference done, start writing results in file')
    #     print(cnt)

    def _infer(self):
        for ii, data in enumerate(self.testloader):
            # print(torch.cuda.memory_allocated(device=self.device))
            if (ii+1) % 100 == 0:
                print('batch {} of {}'.format(ii+1, len(self.testloader)))
            data_input, img_name = data
            output = self._output(data_input)
            self.pred[ii * self.opt.test_batch_size : (ii+1) * self.opt.test_batch_size, :] += output.detach().cpu()

    def _infer_fold(self):
        self._init_dataset()
        self.pred = torch.zeros((len(self.testloader.dataset), 2))

        check_path = self.opt.checkpoints_path
        t1 = time.time()
        for i in range(self.opt.n_fold):
            self.opt.checkpoints_path = os.path.join(check_path, 'checkpoints' + str(i+1))
            print('start inference at {}'.format(self.opt.checkpoints_path))
            self._init_model()
            self._infer()
        t2 = time.time()
        print('total inference time {:.2f} s'.format(t2 - t1))
        self.pred = torch.softmax(self.pred,1).cpu().detach().numpy()[:,1]
        # imgs_name = list()
        # for ii, data in enumerate(self.testloader):
        #     _, img_name = data
        #     imgs_name += list(img_name)
        imgs_name = [img_name.split('\\')[-1].split('.')[0] for img_name in self.testloader.dataset.data_imgs]
        self.csv_data = list(zip(imgs_name, self.pred))
        self.opt.checkpoints_path = check_path

    def _write_submissions(self):
        length_csv = len(self.csv_data)
        with open(self.opt.test_save, 'w') as csvfile:
            csvfile.write('image_name,target\n')
            for i, el in enumerate(self.csv_data):
                a = ','.join([str(el[0]), str(el[1])])
                if i != length_csv-1:
                    csvfile.write(a + '\n')
                else:
                    csvfile.write(a)

    def __call__(self): #, checkpoints_paths):
        try:
            with torch.no_grad():
                self._infer_fold()
                self._write_submissions()
        except:
            print('error occured while infering')
            raise


if __name__ == '__main__':
    TestModel()()
