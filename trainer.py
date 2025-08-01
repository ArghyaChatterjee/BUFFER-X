import gc
from utils.SE3 import *
from utils.timer import Timer, AverageMeter
from loss.desc_loss import ContrastiveLoss, cdist
from tensorboardX import SummaryWriter
from utils.common import make_open3d_point_cloud, ensure_dir


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.cfg = args.cfg
        self.train_modal = self.cfg.stage
        self.epoch = self.cfg.train.epoch
        self.save_dir = args.save_dir
        self.logger = args.logger

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.evaluate_interval = args.evaluate_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        self.desc_loss = ContrastiveLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.Huber_loss = torch.nn.HuberLoss()
        
        # create meters and timers
        self.meter_list = ['desc_loss', 'desc_acc', 'eqv_loss', 'eqv_acc', 'match_loss']
        self.meter_dict = {}
        for key in self.meter_list:
            self.meter_dict[key] = AverageMeter()

    def get_matching_indices(self, source, target, relt_pose, search_voxel_size):
        """
        Input
            - source:     [N, 3]
            - target:     [M, 3]
            - relt_pose:  [4, 4]
        Output:
            - match_inds: [C, 2]
        """
        source = transform(source, relt_pose)
        diffs = source[:, None] - target[None]
        dist = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
        min_ind = torch.cat([torch.arange(source.shape[0])[:, None].cuda(), torch.argmin(dist, dim=1)[:, None]], dim=-1)
        min_val = torch.min(dist, dim=1)[0]
        match_inds = min_ind[min_val < search_voxel_size]

        return match_inds

    def train(self):
        best_loss = 1000000000

        for epoch in range(self.epoch):
            gc.collect()
            self.train_epoch(epoch)

            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate()
                self.logger.info(f'Evaluation: Epoch {epoch}')
                for key in res.keys():
                    self.logger.info(f"{key}: {res[key]}")
                    self.writer.add_scalar(key, res[key], epoch)

                if self.train_modal == 'Desc':
                    if res['desc_loss'] < best_loss:
                        best_loss = res['desc_loss']
                        self._snapshot('best')
                elif self.train_modal == 'Pose':
                    if res['match_loss'] < best_loss:
                        best_loss = res['match_loss']
                        self._snapshot('best')
                else:
                    raise NotImplementedError

            if (epoch + 1) % self.scheduler_interval == 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info('update detector learning rate: %f -> %f' % (old_lr, new_lr))

            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

        # finish all epoch
        self.logger.info("Training finish!... save training results")

    def train_epoch(self, epoch):
        self.logger.info('training start!!')
        self.model.train()
        data_timer, model_timer = Timer(), Timer()

        num_batch = len(self.train_loader)
        num_iter = min(self.cfg.train.max_iter, num_batch)
        data_iter = iter(self.train_loader)
        for i in range(num_iter):
            data_timer.tic()
            data_source = data_iter.__next__()
            data_timer.toc()
            model_timer.tic()

            # forward
            self.optimizer.zero_grad()
            output = self.model(data_source)
            if output is None:
                continue

            if self.train_modal == 'Desc':

                # descriptor loss
                tgt_kpt, src_des, tgt_des = output['tgt_kpt'], output['src_des'], output['tgt_des']
                desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(tgt_kpt, tgt_kpt))

                # equivariant loss to make two cylindrical maps similar
                eqv_loss = self.class_loss(output['equi_score'], output['gt_label'])
                pre_label = torch.argmax(output['equi_score'], dim=1)
                eqv_acc = (pre_label == output['gt_label']).sum() / pre_label.shape[0]

                # refer to RoReg(https://github.com/HpWang-whu/RoReg)
                loss = 4 * desc_loss + eqv_loss
                stats = {
                    "desc_loss": float(desc_loss.item()),
                    "desc_acc": float(accuracy.item()),
                    "eqv_loss": float(eqv_loss.item()),
                    "eqv_acc": float(eqv_acc.item()),
                }

            if self.train_modal == 'Pose':

                pred_ind, gt_ind = output['pred_ind'], output['gt_ind']
                match_loss = self.Huber_loss(pred_ind, gt_ind)
                
                loss = match_loss
                stats = {
                    "match_loss": float(match_loss.item()),
                }

            # backward
            loss.backward()
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            model_timer.toc()
            torch.cuda.empty_cache()

            for key in self.meter_list:
                if stats.get(key) is not None:
                    self.meter_dict[key].update(stats[key])

            if (i + 1) % 200 == 0:
                self.logger.info(f"Epoch: {epoch + 1} [{i + 1:4d}/{num_iter}] "
                      f"data_time: {data_timer.avg:.2f}s "
                      f"model_time: {model_timer.avg:.2f}s ")
                for key in self.meter_dict.keys():
                    self.logger.info(f"{key}: {self.meter_dict[key].avg:.6f}")
                    self.meter_dict[key].reset()
        self._snapshot(f'{epoch}')

    def evaluate(self):
        self.logger.info('validation start!!')
        self.model.eval()
        data_timer, model_timer = Timer(), Timer()

        with torch.no_grad():
            num_batch = len(self.val_loader)
            data_iter = iter(self.val_loader)
            for i in range(num_batch):
                data_timer.tic()
                data_source = data_iter.__next__()
                data_timer.toc()
                model_timer.tic()
                # forward
                output = self.model(data_source)
                if output is None:
                    continue

                if self.train_modal == 'Desc':
                    src_kpt, src_des, tgt_des = output['src_kpt'], output['src_des'], output['tgt_des']
                    desc_loss, diff, accuracy = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))
                    eqv_loss = self.class_loss(output['equi_score'], output['gt_label'])
                    pre_label = torch.argmax(output['equi_score'], dim=1)
                    eqv_acc = (pre_label == output['gt_label']).sum() / pre_label.shape[0]

                    stats = {
                        "desc_loss": float(desc_loss.item()),
                        "desc_acc": float(accuracy.item()),
                        "eqv_loss": float(eqv_loss.item()),
                        "eqv_acc": float(eqv_acc.item()),
                    }

                if self.train_modal == 'Pose':
                    pred_ind, gt_ind = output['pred_ind'], output['gt_ind']
                    match_loss = self.Huber_loss(pred_ind, gt_ind)

                    stats = {
                        "match_loss": float(match_loss.item()),
                    }

                model_timer.toc()
                torch.cuda.empty_cache()
                for key in self.meter_list:
                    if stats.get(key) is not None:
                        self.meter_dict[key].update(stats[key])

        self.model.train()
        res = {}
        for key in self.meter_dict.keys():
            res[key] = self.meter_dict[key].avg

        return res

    def _snapshot(self, info):
        save_path = self.cfg.snapshot_root + f'/{self.train_modal}'
        ensure_dir(save_path)
        torch.save(self.model.module.state_dict(), save_path + f'/{info}.pth')
        self.logger.info(f"Save model to {save_path}/{info}.pth")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
