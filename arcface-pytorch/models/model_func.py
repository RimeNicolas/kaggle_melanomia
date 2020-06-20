import os
import torch

from models.metrics import *
from models.resnet import *

from config.config import Config

def init_model(opt=Config()):
    if opt.backbone == 'resnet18':
        model = resnet18(pretrained=True, num_classes=1000)
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
        metric_fc = ArcMarginProduct(1000, opt.num_classes, s=opt.arc_s, m=opt.arc_m, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    elif opt.metric == 'linear':
        metric_fc = nn.Linear(1000, opt.num_classes, bias=True)
    elif opt.metric == 'None':
        metric_fc = None
    else:
        raise Exception('no known metric was specified')
    return metric_fc

