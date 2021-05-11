# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn.functional as F
from .hook import Hook
from detda.utils.metrics import runningMetric
import time
from detda.models.model_utils import clip_gradient


class LossMetrics(Hook):
    def __init__(self, runner, log_names, group_name, log_interval):
        self.log_interval = log_interval
        self.running_metrics = runningMetric(logger=runner.logger, writer=runner.writer)
        self.running_metrics.add_metrics(log_names, group_name=group_name, metric_type='avgmeter',
                                         log_interval=log_interval)

    def after_train_iter(self, runner):
        batch_output = runner.train_batch_output
        self.running_metrics.update_metrics(batch_output)
        self.running_metrics.log_metrics(runner.iteration)


class LrRecoder(Hook):
    def __init__(self, runner, log_interval):
        self.log_interval = log_interval

    def after_train_iter(self, runner):
        if runner.iteration % self.log_interval == 0:
            log_str = ''
            for name in runner.scheduler_dict:
                temp_lr = runner.scheduler_dict[name].get_last_lr()[0]
                runner.writer.add_scalar('{}/{}'.format('lr', name), temp_lr, runner.iteration)
                log_str += '{}_lr: {:.2e}\t'.format(name, temp_lr)
            runner.logger.info(log_str)


class BackwardUpdate(Hook):
    def __init__(self, runner, ):
        pass

    def before_train_iter(self, runner):
        # optimizer zero grad
        if (runner.iteration - 1) % runner.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if runner.iteration % runner.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                # print('{} step'.format(name))
                runner.optimizer_dict[name].step()
        # scheduler_step
        if runner.iteration % runner.update_iter == 0:
            for name in runner.scheduler_dict.keys():
                runner.scheduler_dict[name].step()


class BackwardUpdatewithAMP(Hook):
    def __init__(self, runner, ):
        pass

    def before_train_iter(self, runner):
        # optimizer zero grad
        if (runner.iteration - 1) % runner.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if runner.iteration % runner.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.scaler.step(runner.optimizer_dict[name])
        # scaler update
        runner.scaler.update()
        # scheduler_step
        if runner.iteration % runner.update_iter == 0:
            for name in runner.scheduler_dict.keys():
                runner.scheduler_dict[name].step()


class TrainTimeRecoder(Hook):
    def __init__(self, runner):
        self.start_time = time.time()
        self.forward_start_time = time.time()
        self.running_metrics = runningMetric(writer=runner.writer, logger=runner.logger)  #
        self.running_metrics.add_metrics('speed', group_name='other', metric_type='avgmeter',
                                         log_interval=runner.log_interval)
        self.running_metrics.add_metrics('forward_speed', group_name='other', metric_type='avgmeter',
                                         log_interval=runner.log_interval)

    def before_train_iter(self, runner):
        self.forward_start_time = time.time()

    def after_train_iter(self, runner):
        self.running_metrics.update_metrics({'other': {'speed': time.time() - self.start_time}})
        self.running_metrics.update_metrics({'other': {'forward_speed': time.time() - self.forward_start_time}})
        self.start_time = time.time()
        self.running_metrics.log_metrics(runner.iteration)


class GradientClipper(Hook):
    def __init__(self, max_num=None):
        self.max_num = max_num

    def after_train_iter(self, runner):
        if runner.iteration % runner.update_iter == 0:
            for name in runner.model_dict.keys():
                clip_gradient(runner.model_dict[name], self.max_num, logger=runner.logger)
