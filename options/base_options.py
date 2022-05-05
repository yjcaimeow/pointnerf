import argparse
import os
from models import find_model_class_by_name
from data import find_dataset_class_by_name
import torch


class BaseOptions:
    def initialize(self, parser: argparse.ArgumentParser):
        #================================ global ================================#
        parser.add_argument('--point_number',
                            type=int,
                            default=1000,
                            help='name of the experiment')
        parser.add_argument('--zoom_in_scale',
                            type=int,
                            default=4,
                            help='name of the experiment')
        parser.add_argument('--scale_factor',
                            type=float,
                            required=10.0,
                            help='name of the experiment')
        parser.add_argument('--filename',
                            type=str,
                            required=True,
                            help='name of the experiment')
        parser.add_argument('--name',
                            type=str,
                            required=True,
                            help='name of the experiment')
        parser.add_argument(
            '--context_weight_norm',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--context_weight_gate',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--context_weight',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--inference',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--fov',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--catWithLocaldir',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--multi_res',
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
        parser.add_argument('--combination',
                            default="concat",
                            type=str,
                            help='cnn | style | condition')
        parser.add_argument('--neural_render',
                            default="style",
                            type=str,
                            help='cnn | style | condition')
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
        parser.add_argument('--vox_res_middle',
                            type=int,
                            default=1000,
                            help='latent_dim')
        parser.add_argument('--network_capacity',
                            type=int,
                            default=16,
                            help='latent_dim')
        parser.add_argument('--serial_batches',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir',
                            type=str,
                            default='./checkpoints',
                            help='models are saved here')
        parser.add_argument('--resume_dir',
                            type=str,
                            default='',
                            help='dir of the previous checkpoint')
        parser.add_argument('--resume_iter',
                            type=str,
                            default='latest',
                            help='which epoch to resume from')
        parser.add_argument('--z_dim',
                            type=int,
                            default=256,
                            help='latent_dim')
        parser.add_argument('--frames_length',
                            type=int,
                            default=30,
                            help='frames length')
        parser.add_argument('--N_importance',
                            type=int,
                            default=128,
                            help='input batch size')
        parser.add_argument('--N_samples',
                            type=int,
                            default=64,
                            help='input batch size')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='input batch size')
        parser.add_argument('--render_only',
                            type=int,
                            default=0,
                            help='1 for render_only dataset')
        parser.add_argument('--show_tensorboard',
                            type=int,
                            default=0,
                            help='plot loss curves with tensorboard')
        parser.add_argument('--contain_coarse',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--half_supervision',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--pe_bound',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--weight_norm',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--alpha_as_weight',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--proposal_nerf',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--unified',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--nerf_raycolor',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--only_nerf',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--nerf_distill',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--nerf_aug',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--ddp_train',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--mask_moving_obj',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--multi_nerf',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--inference_use_nerf',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--debug',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--prune_points',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--nerf_create_points',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--iter_pg',
                            type=int,
                            default=100,
                            help='feed batches in order without shuffling')
        parser.add_argument('--seq_num',
                            type=int,
                            default=5,
                            help='feed batches in order without shuffling')
        parser.add_argument('--sample_num',
                            type=int,
                            default=5000,
                            help='feed batches in order without shuffling')
        parser.add_argument('--port',
                            type=str,
                            default='12345',
                            help='feed batches in order without shuffling')
        parser.add_argument('--world_size',
                            type=int,
                            default=1,
                            help='feed batches in order without shuffling')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        parser.add_argument('--local_rank',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
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

        opt, _ = parser.parse_known_args()

        model_name = opt.model
        find_model_class_by_name(model_name).modify_commandline_options(
            parser, self.is_train)

        dataset_name = opt.dataset_name
        if dataset_name is not None:
            find_dataset_class_by_name(
                dataset_name).modify_commandline_options(
                    parser, self.is_train)

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

        self.print_and_save_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [
            int(x) for x in opt.gpu_ids.split(',') if x.strip() and int(x) >= 0
        ]
#        if len(opt.gpu_ids) > 0:
#            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
