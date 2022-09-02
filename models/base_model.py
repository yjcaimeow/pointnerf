import torch
from torch import nn
import os
from .helpers.networks import get_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cprint import *

class BaseModel:
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return self.__class__.__name__

    def initialize(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

        if self.opt.ddp_train:
            self.device = torch.device('cuda:{}'.format(self.local_rank))
        else:
            self.device = torch.device('cuda:{}'.format(0))
            cprint.err(self.device)

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.backends.cudnn.benchmark = True

        self.loss_names = []  # losses to report
        self.model_names = []  # models that will be used
        self.visual_names = []  # visuals to show at test time

    def set_input(self, input: dict):
        self.input = input


    def set_ddp(self):
        for name in self.model_names:
            assert isinstance(name, str)
            net = getattr(self, 'net_{}'.format(name))
            assert isinstance(net, nn.Module)
            if self.opt.ddp_train:
                if self.opt.sync_bn:
                    net = nn.SyncBatchNorm.convert_sync_batchnorm(net.to(self.device))
                    net = DDP(net, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
                else:
                    net = DDP(net.to(self.device), device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            else:
                net = torch.nn.DataParallel(net.cuda())
            setattr(self, 'net_{}'.format(name), net)

    def forward(self):
        '''Run the forward pass. Read from self.input, set self.output'''
        raise NotImplementedError()

    def setup(self, opt):
        '''Creates schedulers if train, Load and print networks if resume'''
        if self.is_train:
            self.schedulers = [
                get_scheduler(optim, opt) for optim in self.optimizers
            ]
        if not self.is_train or opt.resume_dir:
            print("opt.resume_iter!!!!!!!!!", opt.resume_iter)
            self.load_networks(opt.resume_iter)
        self.print_networks(opt.verbose)

    def eval(self):
        '''turn on eval mode'''
        for net in self.get_networks():
            net.eval()

    def train(self):
        for net in self.get_networks():
            net.train()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_networks(self) -> [nn.Module]:
        ret = []
        for name in self.model_names:
            assert isinstance(name, str)
            net = getattr(self, 'net_{}'.format(name))
            assert isinstance(net, nn.Module)
            ret.append(net)
        return ret

    def get_current_visuals(self, data=None, no_extra=False):
        ret = {}
        for name in self.visual_names:
            assert isinstance(name, str)
            if name not in ["gt_image_ray_masked", "ray_depth_masked_gt_image", "ray_depth_masked_coarse_raycolor", "ray_masked_coarse_raycolor"]:
                if no_extra:
                    if name not in ["sample_loc", "sample_loc_w", "ray_valid", "decoded_features"]:
                        ret[name] = getattr(self, name)
                else:
                    ret[name] = getattr(self, name)
        if "coarse_raycolor" not in self.visual_names:
            ret["coarse_raycolor"] = getattr(self, "coarse_raycolor")
        return ret

    def get_current_losses(self):
        ret = {}
        for name in self.loss_names:
            assert isinstance(name, str)
            ret[name] = getattr(self, 'loss_' + name)
        return ret

    def save_networks(self, iteration, epoch, other_states={}, back_gpu=True):
        for name, net in zip(self.model_names, self.get_networks()):
            save_filename = '{}_{}_net_{}.pth'.format(epoch, os.getpid(), name)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(net.state_dict(), save_path)
            #try:
            #    if isinstance(net, nn.DataParallel):
            #        net = net.module
            #    net.cpu()
            #    torch.save(net.state_dict(), save_path)
            #    if back_gpu:
            #        net.cuda()
            #except Exception as e:
            #    print("savenet:", e)

        save_filename = '{}_{}_states.pth'.format(epoch, os.getpid())
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(other_states, save_path)

    def load_networks(self, epoch):
        for name, net in zip(self.model_names, self.get_networks()):
            print('loading', name)
            assert isinstance(name, str)
            load_filename = '{}_net_{}.pth'.format(epoch, name)
            load_path = os.path.join(self.opt.resume_dir, load_filename)

            if not os.path.isfile(load_path):
                print('cannot load', load_path)
                continue

            state_dict = torch.load(load_path, map_location='cpu')
            #state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(net, nn.DataParallel):
                net = net.module

            net.load_state_dict(state_dict, strict=False)


    def print_networks(self, verbose):
        print('------------------- Networks -------------------')
        for name, net in zip(self.model_names, self.get_networks()):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network {}] Total number of parameters: {:.3f}M'.format(
                name, num_params / 1e6))
        print('------------------------------------------------')

    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step()
        if int(os.environ["LOCAL_RANK"])!=0:
            return
        for i, optim in enumerate(self.optimizers):
            lr = optim.param_groups[0]['lr']
            if "opt" in kwargs:
                opt = kwargs["opt"]
                if not opt.lr_policy.startswith("iter") or \
                        ("total_steps" in kwargs and kwargs["total_steps"] % opt.print_freq == 0):
                    print('optimizer {}, learning rate = {:.7f}'.format(i + 1, lr))
            else:
                print('optimizer {}, learning rate = {:.7f}'.format(i + 1, lr))


