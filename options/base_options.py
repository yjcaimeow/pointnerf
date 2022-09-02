import argparse
import os
from models import find_model_class_by_name
from data import find_dataset_class_by_name
import torch
import yaml

class BaseOptions:
    def initialize(self, parser: argparse.ArgumentParser):
        #================================ global ================================#
        parser.add_argument(
                '-c',
                '--config_yaml',
                default=
                './configs/train.yaml',
                type=str,
                metavar='FILE',
                help='YAML config file specifying default arguments')
        parser.add_argument('--name',
                            type=str,
                            default='exp',
                            help='name of the experiment')
        parser.add_argument('--only_render',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--all_frames',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--ddp_train',
                            action='store_true',
                            help='indicate a debug run')

        parser.add_argument('--PROCESSNUM',
                type=int,
                default=4,
                help='local_rank init value')
        parser.add_argument('--port',
                type=int,
                default=16666,
                help='local_rank init value')
        parser.add_argument('--local_rank',
                type=int,
                default=0,
                help='local_rank init value')
        parser.add_argument('--world_size', default=1, type=int,
                                                 help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

        parser.add_argument('--sync_bn',
                            type=int,
                            default=1,
                            help='1 for render_only dataset')
        parser.add_argument(
            '--clip_knn',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--relative_thresh',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--timestamp',
            action='store_true',
            help='suffix the experiment name with current timestamp')
        #================================ model aggregator========================#
        parser.add_argument(
            '--pseudo_visual_items',
            type=str,
            default="aaaaa",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--pseudo_gt_load_type',
            type=str,
            default="online",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--embed_init_type',
            type=str,
            default="model",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--loss_type',
            type=str,
            default="l1",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--perceiver_io_type',
            type=str,
            default="each_sample_loc",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--attention_type',
            type=str,
            default="normal",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
                        '--ddp_init_type',
                        type=str,
                        default="old",
                        help='name of dataset, determine which dataset class to use')
        parser.add_argument(
                        '--k_type',
                        type=str,
                        default="voxel",
                        help='name of dataset, determine which dataset class to use')
        parser.add_argument('--scans',
                            type=str,
                            nargs='+',
                            default=("none "),
                            help='saving frequency')
        parser.add_argument(
            '--load_type',
            type=str,
            default="ceph_not",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--agg_type',
            type=str,
            default="mlp",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--load_init_pcd_type',
            type=str,
            default="grid",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--ray_dir_type',
            type=str,
            default="global",
            help='name of dataset, determine which dataset class to use')
        parser.add_argument('--all_sample_loc',type=int,default=0,help='1 for render_only dataset')
        parser.add_argument('--progressive_distill',type=int,default=0,help='1 for render_only dataset')
        parser.add_argument('--embed_color',type=int,default=0,help='1 for render_only dataset')
        parser.add_argument('--gap', type=float, default=0.2, help='name of the experiment')
        parser.add_argument('--knn_k', type=int, default=8, help='name of the experiment')
        parser.add_argument('--light_N', type=int, default=8, help='name of the experiment')
        parser.add_argument('--light_D', type=int, default=290, help='name of the experiment')
        parser.add_argument('--light_C', type=int, default=290, help='name of the experiment')
        parser.add_argument('--light_num_self_attention_heads', type=int, default=2, help='name of the experiment')
        parser.add_argument('--light_num_self_attention_blocks', type=int, default=2, help='name of the experiment')
        parser.add_argument('--light_num_self_attention_layers_per_block', type=int, default=2, help='name of the experiment')
        parser.add_argument('--num_perceiver_io_freqs', default=4, type=int, help='# threads for loading data')
        #================================ dataset ================================#
        parser.add_argument('--data_root',
                            type=str,
                            default=None,
                            help='path to the dataset storage')
        parser.add_argument(
            '--dataset_name',
            type=str,
            default=None,
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=float("inf"),
            help='Maximum number of samples allowed per dataset.'
            'If the dataset directory contains more than max_dataset_size, only a subset is loaded.'
        )
        parser.add_argument('--n_threads',
                            default=1,
                            type=int,
                            help='# threads for loading data')
        #================================ MVS ================================#

        parser.add_argument('--geo_cnsst_num',
                            default=2,
                            type=int,
                            help='# threads for loading data')


        #================================ model ================================#
        parser.add_argument('--bgmodel',
                            default="No",
                            type=str,
                            help='No | sphere | plane')

        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='name of model, determine which network model to use')

        #================================ running ================================#
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='input batch size')
        parser.add_argument('--render_only',
                            type=int,
                            default=0,
                            help='1 for render_only dataset')
        parser.add_argument('--serial_batches',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
        parser.add_argument('--checkpoints_dir',
                            type=str,
                            default='./checkpoints',
                            help='models are saved here')
        parser.add_argument('--show_tensorboard',
                            type=int,
                            default=0,
                            help='plot loss curves with tensorboard')
        parser.add_argument('--resume_dir',
                            type=str,
                            default='',
                            help='dir of the previous checkpoint')
        parser.add_argument('--resume_iter',
                            type=str,
                            default='latest',
                            help='which epoch to resume from')
        parser.add_argument('--debug',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--vid',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
        parser.add_argument('--resample_pnts',
                            type=int,
                            default=-1,
                            help='resample the num. initial points')
        parser.add_argument('--inall_img',
                            type=int,
                            default=1,
                            help='all points must in the sight of all camera pose')
        parser.add_argument('--test_train', type=int, default=0, help='test on training set for debugging')

        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt, remaining = parser.parse_known_args()

        model_name = opt.model
        find_model_class_by_name(model_name).modify_commandline_options(
            parser, self.is_train)

        dataset_name = opt.dataset_name
        if dataset_name is not None:
            find_dataset_class_by_name(
                dataset_name).modify_commandline_options(
                    parser, self.is_train)

        ### load from config
        if opt.config_yaml:
            with open(os.path.join(os.getcwd(), '../', opt.config_yaml), 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        self.parser = parser

        return parser.parse_args()

    def print_and_save_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # if opt.is_train:
        #     expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # else:
        #     expr_dir = os.path.join(opt.resume_dir, opt.name)
        opt.name = opt.name[20:]
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        if opt.timestamp:
            import datetime
            now = datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S')
            opt.name = opt.name + '_' + now
        sstr = "-".join(opt.scans)
        #opt.name = opt.name+'.'+sstr
        opt.name = opt.name+'.'+sstr+'.gpu'+str(opt.PROCESSNUM)

        self.print_and_save_options(opt)

        self.opt = opt
        return self.opt
