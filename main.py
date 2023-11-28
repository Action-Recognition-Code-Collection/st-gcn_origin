#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    # processors字典的键为运行方式的名称，值为对应的执行类
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    # 给subparser添加各个具体解析器——recognition、demo_old、demo与demo_offline
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        # 从各执行类的get_parser()中得到对应的解析参数
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    # 根据subparser得到的具体解析器名称获得对应执行类
    Processor = processors[arg.processor]
    # subparser名称（不含）之后的参数全部传入执行类的构造函数
    p = Processor(sys.argv[2:])

    p.start()
