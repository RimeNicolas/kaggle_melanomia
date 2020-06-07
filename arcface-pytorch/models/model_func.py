import os
import torch

from models.metrics import *
from models.resnet import *

from config.config import Config

def init_model(opt=Config()):
    if opt.backbone == 'resnet18':
        model = resnet18(num_classes=2)
    elif opt.backbone == 'resnet34':
        model = resnet34(num_classes=2)
    elif opt.backbone == 'resnet50':
        model = resnet50(num_classes=2)
    elif opt.backbone == 'resnet152':
        model = resnet152()
    else:
        raise Exception('no known model was specified')
    return model


def init_metric(opt=Config()):
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    elif opt.metric == 'None':
        metric_fc = None
    else:
        raise Exception('no known metric was specified')
    return metric_fc


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name