# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sys
module_path = os.path.abspath(os.path.join('./'))
sys.path.append(module_path)

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0
best_major = 0
best_minor = 0


def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    return list(class_num_list)


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return label_freq_array


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class Entropy(nn.Module):
    def __init__(self,args,temperature):
        super(Entropy, self).__init__()
        self.temperature = temperature
        self.args = args
    def forward(self, x):
        target = torch.ones((x.size(0),self.args.num_classes)).float().to(self.args.device) / self.args.num_classes
        
        b = (F.softmax(x / self.temperature, dim=1) - target)**2
        b = 0.5 * b.sum(dim=1).mean()
        return b

class Entropy_revision(nn.Module):
    def __init__(self,args,temperature):
        super(Entropy_revision, self).__init__()
        self.temperature = temperature
        self.args = args
        target = torch.ones((1,self.args.num_classes)).float().to(self.args.device) / self.args.num_classes
        self.entMax = -1 * target * torch.log(target)
    def forward(self, x):
        b = -1 * F.softmax(x / self.temperature, dim=1) * F.log_softmax(x / self.temperature, dim=1)
        b = self.entMax.sum(dim=1) - b.sum(dim=1)
        b = b.mean()
        return b


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labelled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        help='whether to reverse the distribution of the unlabeled data')
    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau1', default=2, type=float,
                        help='tau for head1 consistency')
    parser.add_argument('--tau12', default=2, type=float,
                        help='tau for head2 consistency')
    parser.add_argument('--tau2', default=2, type=float,
                        help='tau for head2 balanced CE loss')
    parser.add_argument('--ema-u', default=0.9, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--est-epoch', default=5, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')
    parser.add_argument('--debiasstart', type=int, default=100, help='Long tailed or step imbalanced')
    parser.add_argument('--entropy-weights',default=1.0,type=float)
    parser.add_argument('--optim', default='sgd', type=str,
                        choices=['sgd','adam'],)
    parser.add_argument('--path', default='./public_dataset/', type=str,
                        )
    args = parser.parse_args()
    global best_acc
    global best_acc_b
    global best_major
    global best_minor

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    if args.seed is not None:
        set_seed(args)
    args.device = device

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(args.out+'information.log')
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}",)

    logger.info(dict(args._get_kwargs()))

    

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar-100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.dataset_name = 'svhn'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
        labeled_dataset, unlabeled_dataset, test_dataset, _ , l_samples , u_samples= DATASET_GETTERS[args.dataset](
            args, args.path + args.dataset_name)
        logger.info(f'Distribution on Labeled Samples : {l_samples}' )
        logger.info(f'Distribution on UnLabeled Samples : {u_samples}' )
    elif args.dataset == 'stl10':
        labeled_dataset, unlabeled_dataset, test_dataset, _ , l_samples= DATASET_GETTERS[args.dataset](
            args, args.path + args.dataset_name)
        logger.info(f'Distribution on Labeled Samples : {l_samples}' )
    
    elif args.dataset == 'smallimagenet':
        labeled_dataset, unlabeled_dataset, test_dataset, _ = DATASET_GETTERS[args.dataset](
            args, args.path + args.dataset_name)
    

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    

    args.taumin = 0
    args.taumax = args.tau1

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch(ABC)-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(['Top1 acc', 'Top5 acc', 'Best Top1 acc', 'major','minor','best_major','best_minor'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optim =='sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                            momentum=0.9, nesterov=args.nesterov)
    elif args.optim=='adam':
        optimizer = optim.Adam(grouped_parameters, lr=args.lr)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    summary(model.cuda(),(3,32,32))
    exit(1)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.u_py = checkpoint['u_py']
        args.u_py = args.u_py.to(args.device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)
    args.logger.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    global best_major
    global best_minor
    test_accs = []
    avg_time = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)


    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch+1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_entropy = AverageMeter()
        mask_probs = AverageMeter()
        current_time = time.time()
       

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, _), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), u_real = next(unlabeled_iter)


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            
            targets_x = targets_x.to(args.device)

            
            logits_feat = model(inputs)
            logits = model.classify(logits_feat)

            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits
            
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            
                
            pseudo_label_target = torch.softmax(logits_u_w.detach() / args.T, dim=-1)


            max_probs, targets_u = torch.max(pseudo_label_target, dim=-1)

            targets_u2 = torch.zeros(batch_size*args.mu, args.num_classes).to(args.device).scatter_(1, targets_u.view(-1,1), 1).to(args.device).to(args.device).detach()

            mask = max_probs.ge(args.threshold) # 0.95
            # mask = torch.cat([mask,mask],0)
          
            
            mask = mask.float()


            logits_u_s_twice = logits_u_s
            targets_u_twice = targets_u2
            # targets_u_twice = torch.cat([pseudo_label_target,pseudo_label_target],dim=0)
            
            Lu = -torch.mean(torch.sum(torch.log(F.softmax(logits_u_s_twice/args.T, dim=1)+1e-8) * targets_u_twice, dim=1) * mask)

            loss = Lx + Lu
            
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_entropy.update(0)
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
        print(f'spending time = {time.time()-current_time}')
        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_top5_acc, accperclass = test(args, test_loader, test_model, epoch)

            if args.dataset == 'cifar10':
                major = accperclass[:5].mean()
                minor = accperclass[5:].mean()
            elif args.dataset == 'svhn':
                major = accperclass[:5].mean()
                minor = accperclass[5:].mean()
            elif args.dataset == 'stl10':
                major = accperclass[:5].mean()
                minor = accperclass[5:].mean()
            elif args.dataset == 'cifar100':
                major = accperclass[:50].mean()
                minor = accperclass[50:].mean()
            
            major = major * 100
            minor = minor * 100
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/2.test_acc_major', major, epoch)
            args.writer.add_scalar('test/3.test_acc_minor', minor, epoch)
            args.writer.add_scalar('test/4.test_loss', test_loss, epoch)
            
            is_best = test_acc > best_acc_b

            best_acc = max(test_acc, best_acc)
            best_acc_b = max(test_acc, best_acc_b)
            if test_acc >= best_acc_b:
                best_major = major
                best_minor = minor
                
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch+1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out, epoch_p=epoch+1)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc,major,minor,best_major,best_minor])
            logger.info(f'train_loss : {losses.avg:.4f} | loss_x : {losses_x.avg:.4f} | loss_u : {losses_u.avg:.4f}')
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    accperclass = np.zeros((args.num_classes))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_feat = model(inputs)
            outputs = model.classify(outputs_feat)
            loss = F.cross_entropy(outputs, targets)

            targetsonehot = torch.zeros(inputs.size()[0], args.num_classes).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            unbiased=torch.argmax(outputs,dim=1)
            outputs2onehot = torch.zeros(inputs.size()[0], args.num_classes).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)
            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)
            
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))

    if args.dataset=='cifar10':
        accperclass=accperclass/1000
    elif args.dataset=='svhn':
        accperclass=accperclass/1500
    elif args.dataset=='cifar100':
        accperclass=accperclass/100
    elif args.dataset=='stl10':
        accperclass=accperclass/800

    return losses.avg, top1.avg, top5.avg, accperclass


if __name__ == '__main__':
    main()
