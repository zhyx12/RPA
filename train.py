import os
import shutil
import torch
import random
import argparse
import numpy as np
from clsda.utils import get_root_logger, get_root_writer
from clsda.loader import parse_args_for_dataset
from clsda.models import parse_args_for_models
from clsda.utils.utils import deal_with_val_interval
#
from PIL import ImageFile
from clsda.utils.utils import move_models_to_gpu
import time
from clsda.runner.hooks import LrRecorder, TrainTimeRecoder, SaveModel, SchedulerStep
from mmcv import Config
from clsda.trainers import build_trainer, build_validator
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(cfg, logger, logdir, args):
    #
    control_cfg = cfg['control']
    # torch vision
    logger.info('torch vision is {}'.format(torch.__version__))
    # Setup random seeds
    if 'seed' in control_cfg:
        random_seed = cfg['control'].get('seed', None)
    else:
        random_seed = None
    if random_seed is None:
        random_seed = random.randint(1000, 2000)
    logger.info("Random Seed is {}".format(random_seed))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # debug mode: set dataset sample number
    debug_flag = args.debug
    train_debug_sample_num = args.train_debug_sample_num
    test_debug_sample_num = args.test_debug_sample_num
    # debug mode: change log_interval和val_interval
    if debug_flag:
        control_cfg['log_interval'] = args.debug_log_interval
        control_cfg['val_interval'] = args.debug_val_interval
    # cuda_flag
    cuda_flag = (not args.no_cuda) and torch.cuda.is_available()
    #
    # build dataloader
    train_loaders, test_loaders = parse_args_for_dataset(cfg['datasets'], debug=debug_flag,
                                                         train_debug_sample_num=train_debug_sample_num,
                                                         test_debug_sample_num=test_debug_sample_num,
                                                         random_seed=random_seed, data_root=args.data_root,
                                                         task_type=args.task_type)
    for i, loader in enumerate(train_loaders):
        logger.info('{} train loader has {} images'.format(i, len(loader.dataset)))
    # build model and corresponding optimizer, scheduler
    n_classes = train_loaders[0].dataset.n_classes
    logger.info('Trainer class is {}'.format(args.trainer))
    model_related_results = parse_args_for_models(cfg['models'], task_type=args.task_type, n_classes=n_classes)
    model_dict, optimizer_dict, scheduler_dict, device_dict = model_related_results
    # move model to gpu
    if cuda_flag:
        model_dict = move_models_to_gpu(model_dict, device_dict)
    # cudnn settings
    torch.backends.cudnn.enabled = True
    if control_cfg['cudnn_deterministic']:
        logger.info('Using cudnn deterministic model')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    #
    # gather trainer args
    training_args = cfg['train']
    training_args.update({
        'type': args.trainer,
        'cuda': cuda_flag,
        'model_dict': model_dict,
        'optimizer_dict': optimizer_dict,
        'scheduler_dict': scheduler_dict,
        'train_loaders': train_loaders,
        'logdir': logdir,
        'log_interval': control_cfg['log_interval']
    })
    #
    pretrained_model = control_cfg.get('pretrained_model', None)
    checkpoint_file = control_cfg.get('checkpoint', None)
    # build trainer
    trainer = build_trainer(training_args)
    trained_iteration = 0
    # load pretrained weights
    if pretrained_model is not None:
        if '~' in pretrained_model:
            pretrained_model = os.path.expanduser(pretrained_model)
        assert os.path.isfile(pretrained_model), '{} is not a weight file'.format(pretrained_model)
        logger.info('Load pretrained model in {}'.format(pretrained_model))
        trainer.load_pretrained_model(pretrained_model)
    # resume training from checkpoint
    if checkpoint_file is not None:
        if '~' in checkpoint_file:
            checkpoint_file = os.path.expanduser(checkpoint_file)
        trainer.resume_training(checkpoint_file)
        trained_iteration = trainer.get_trained_iteration_from_scheduler()
    #
    # build validator
    test_args = cfg['test']
    test_args.update(
        {
            'type': args.validator,
            'cuda': cuda_flag,
            'model_dict': model_dict,
            'test_loaders': test_loaders,
            'logdir': logdir,
        }
    )
    validator = build_validator(test_args)
    ########################################
    log_interval = control_cfg['log_interval']
    updater_iter = control_cfg.get('update_iter', 1)
    # 注册训练的hook
    lr_recoder = LrRecorder(log_interval)
    train_time_recoder = TrainTimeRecoder(log_interval)
    save_model_hook = SaveModel(control_cfg['max_save_num'], save_interval=control_cfg['save_interval'])
    scheduler_step = SchedulerStep(updater_iter)
    trainer.register_hook(lr_recoder, priority='HIGH')
    trainer.register_hook(train_time_recoder)
    trainer.register_hook(save_model_hook,
                          priority='LOWEST')  # save model after scheduler step to get right the iteration number
    trainer.register_hook(scheduler_step, priority='VERY_LOW')

    # 处理val_interval
    val_point_list = deal_with_val_interval(control_cfg['val_interval'], max_iters=control_cfg['max_iters'],
                                            trained_iteration=trained_iteration)
    # 训练和测试交替的流程
    last_val_point = trained_iteration
    for val_point in val_point_list:
        # 训练
        trainer(train_iteration=val_point - last_val_point)
        time.sleep(2)
        # 测试
        save_flag = validator(trainer.iteration)
        #
        if save_flag:
            save_path = os.path.join(trainer.logdir, "best_model.pth".format(trainer.iteration))
            torch.save(trainer.state_dict(), save_path)
        #
        last_val_point = val_point
        # 清理显存
        torch.cuda.empty_cache()
    #
    # save_flag = validator(trainer.iteration)


if __name__ == "__main__":
    project_root = os.getcwd()
    package_name = 'clsda'
    #
    # trainer_name = 'fixmatch'
    # config_path = 'configs/fixmatch/fixmatch_officehome_A_C_baseline.py'
    # trainer_name = 'episodetrain'
    # config_path = 'configs/episode_training/episoed_training_officehome_A_C_test.py'
    # trainer_name= 'mme'
    # config_path = 'configs/mme/mme_officehome_A_C_double_aug_test.py'
    # config_path = 'configs/mme/mme_officehome_A_C_weights_temp_1e0.py'
    trainer_name = 'barlowtwins'
    config_path = 'configs/barlow_twins/barlow_twins_rand_aug_officehome_A_C.py'
    # trainer_name = 'multiviewadv'
    # config_path = 'configs/multiview_adv/multiview_adv_training_officehome_A_C_min_entropy_test.py'
    #
    # trainer_name = 'classwisealign'
    # config_path = 'configs/classwise_align/classwise_align_officehome_A_C_test.py'
    #
    # trainer_name = 'pseudolabel'
    # config_path = 'configs/barlow_twins/barlow_twins_officehome_A_C_test_with_predictor.py'
    #
    # trainer_name = 'ent'
    # config_path = 'configs/ent/ent_officehome_A_C_3e1_test.py'
    #
    data_root = os.path.join(project_root, 'data')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--job_id', default='debug')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--train_debug_sample_num', type=int, default=10)
    parser.add_argument('--test_debug_sample_num', type=int, default=10)
    parser.add_argument('--debug_log_interval', type=int, default=1)
    parser.add_argument('--debug_val_interval', type=int, default=8)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--trainer', help='trainer classes', default=trainer_name)
    parser.add_argument('--validator', help='validator classes', default=trainer_name)
    parser.add_argument('--data_root', help='dataset root path', default=data_root)
    parser.add_argument('--task_type', help='segmentation or detection', default="cls")
    parser.add_argument('--log_level', help='logging level', default=logging.INFO)
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default=project_root + "/" + config_path,
        help="Configuration file to use"
    )

    args = parser.parse_args()
    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-3],
                          'job_' + args.job_id + '_exp_' + str(run_id))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    #
    shutil.copy(args.config, logdir)  #
    shutil.copytree('./{}'.format(package_name), os.path.join(logdir, 'source_code'))
    #
    cfg = Config.fromfile(args.config)
    predefined_keys = ['datasets', 'models', 'control', 'train', 'test']
    old_keys = list(cfg._cfg_dict.keys())
    for key in old_keys:
        if not key in predefined_keys:
            del cfg._cfg_dict[key]
    cfg_save_path = os.path.join(logdir, 'config.py')
    cfg.dump(cfg_save_path)
    #
    timestamp = time.strftime('runs_%Y_%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(logdir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=args.log_level)
    logger.info('log dir is {}'.format(logdir))
    logger.info('Let the games begin')
    logger.info('Job ID in Cluster is {}'.format(args.job_id))
    #
    tb_writer = get_root_writer(log_dir=logdir)
    #
    train(cfg, logger, logdir, args)
