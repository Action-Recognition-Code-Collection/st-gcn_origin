#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

class IO():
    """
        IO Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # 获取命令行参数
        p = parser.parse_args(argv)
        # 命令行参数中存在-c/--config，则尝试从配置文件加载参数
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            # 获得parser中预定义的所有参数名
            key = vars(p).keys()
            # 将配置文件default_arg中的配置项与parser中定义的参数比对，确保配置文件中的配置项均在
            # parser中有定义
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            # 将配置文件中的参数都放入parser中
            parser.set_defaults(**default_arg)

        # 再次解析命令行参数，命令行中指定的参数会覆盖配置文件中的参数，同时也对配置文件中未指定的
        # 进行补充
        self.arg = parser.parse_args(argv)

    def init_environment(self):
        # 根据参数设定work_dir(str), save_log(bool), print_log(bool)
        self.io = torchlight.IO(
            self.arg.work_dir,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log)
        # 将所有参数保存保存在work_dir/config.yaml中
        self.io.save_arg(self.arg)

        # gpu
        if self.arg.use_gpu:
            # gpu编号写入环境变量CUDA_VISIBLE_DEVICES，返回的是List[int]
            gpus = torchlight.visible_gpu(self.arg.device)
            # torch将一个0写入gpu，使得程序在nvidia-smi中可以看到
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        # 根据--model指定的参数加载模型类，--model_args中以dict指定的参数作为实例化模型的参数给入
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights)

    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev)
        # 将当前类所有以torch.nn.modules开头的属性、方法均移动到gpu
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        # model parallel
        # 若有多块gpu，则将模型数据并行到各gpu上
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable

        return parser
