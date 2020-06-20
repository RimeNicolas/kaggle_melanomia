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

    def _init_model(self):
        self.model = init_model(self.opt)
        path_parameters = os.path.join(self.opt.checkpoints_path, self.opt.path_model_parameters_test)
        self.model.load_state_dict(torch.load(path_parameters))
        self.model.to(self.device)
        self.model.eval()

        self.metric_fc = init_metric(self.opt)
        path_parameters = os.path.join(self.opt.checkpoints_path, self.opt.path_metric_parameters_test)
        self.metric_fc.load_state_dict(torch.load(path_parameters))
        self.metric_fc.to(self.device)
        self.metric_fc.eval()
        
    def init_dataset(self):
        test_set = Dataset(self.opt.test_list, phase='test', input_shape=self.opt.input_shape)

        self.testloader = torch.utils.data.DataLoader(test_set,
                                    batch_size=self.opt.test_batch_size,
                                    shuffle=False,
                                    num_workers=self.opt.num_workers)

    def _output(self, data_input):
        data_input = data_input.to(self.device)
        feature = self.model(data_input)
        output = self.metric_fc(feature)
        # return torch.argmax(output, axis=1)
        return torch.softmax(output, 1)[:,1]

    def _infer_old(self):
        self.t0 = time.time()
        cnt = Counter()
        print('start testing')
        for ii, data in enumerate(self.testloader):
            # if ii > 1:
            #     break
            if (ii+1) % 100 == 0:
                print('batch {} of {}'.format(ii+1, len(self.testloader)))
            data_input, img_name = data
            output = self._output(data_input)
            output = list(output.detach().cpu().numpy())
            cnt += Counter(np.round(output))
            self.csv_data += zip(img_name, output)
        self.csv_data = sorted(self.csv_data, key = lambda x : x[0].split('.')[0])
        print('inference done, start writing results in file')
        print(cnt)

    # def _infer(self):
    #     print('start testing')
    #     for ii, data in enumerate(self.testloader):
    #         # if ii > 1:
    #         #     break
    #         if (ii+1) % 100 == 0:
    #             print('batch {} of {}'.format(ii+1, len(self.testloader)))
    #         data_input, img_name = data
    #         output = self._output(data_input)
    #         output = list(output.detach().cpu().numpy())

    def __call__(self): #, checkpoints_paths):
        self.init_dataset()
        self._init_model()
        self.csv_data = list()
        self._infer_old()

        length_csv = len(self.csv_data)
        print('length of csv data :', length_csv)
        with open(self.opt.test_save, 'w') as csvfile:
            csvfile.write('image_name,target\n')
            for i, el in enumerate(self.csv_data):
                a = ','.join([str(el[0]), str(el[1])])
                if i != length_csv-1:
                    csvfile.write(a + '\n')
                else:
                    csvfile.write(a)
        print('done')

if __name__ == '__main__':
    TestModel()()
