import os
import yaml
import shutil
import torch
import random
import argparse
import numpy as np

from detda.utils.utils import get_logger
from detda.loader import parse_args_for_dataset
from detda.models import parse_args_for_models
from detda.trainers import get_trainer

#
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(cfg, logger, logdir, args):
    # TODO: 将来可能会去掉
    # torch.cuda.set_device(0)
    # 输出torch的版本号
    logger.info('torch vision is {}'.format(torch.__version__))
    # Setup seeds
    # 产生随机数并且记录下来
    if 'control' in cfg:
        random_seed = cfg['control'].get('random_seed', None)
    else:
        random_seed = None
    if random_seed is None:
        random_seed = random.randint(1000, 2000)
    logger.info("Random Seed is {}".format(random_seed))
    print('random seed is {}'.format(random_seed))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    debug = args.debug
    train_debug_sample_num = args.train_debug_sample_num
    test_debug_sample_num = args.test_debug_sample_num
    trainer_class = get_trainer(args.trainer)
    cuda_flag = not args.no_cuda

    # get dataloader
    train_loaders, test_loaders = parse_args_for_dataset(cfg['dataset'], debug=debug, logger=logger,
                                                         train_debug_sample_num=train_debug_sample_num,
                                                         test_debug_sample_num=test_debug_sample_num,
                                                         random_seed=random_seed, data_root=args.data_root,
                                                         task_type=args.task_type)
    # Setup Model
    n_classes = train_loaders[0].dataset.n_classes
    logger.info('Trainer class is {}'.format(args.trainer))
    if args.trainer in ['fullsupervisedcyc', 'separableadv']:
        model_dict, optimizer_dict, scheduler_dict = parse_args_for_single_cyc_models(cfg['model'], n_classes, logger)
        device_dict = None
    elif args.trainer in ['tripletclassifier', 'doublecross', 'tripletclassifierold23', 'tripletclassifierold211',
                          'tripletclassifierold225', 'crossmodeladvwithtransform', 'ssl']:
        model_dict, optimizer_dict, scheduler_dict, device_dict = parse_args_for_double_transformer_models(cfg['model'],
                                                                                                           n_classes,
                                                                                                           logger)
    else:
        model_dict, optimizer_dict, scheduler_dict, device_dict = parse_args_for_models(cfg['model'], n_classes, logger,
                                                                                        task_type=args.task_type)

    # model_1 = model_dict['vgg_base_model']
    # model_2 = model_dict['res_base_model']
    # path_1 = './model_1.pth'
    # path_2 = './model_2.pth'
    # torch.save(model_1.state_dict(), path_1)
    # torch.save(model_2.state_dict(), path_2)
    # exit(0)

    training_flag = cfg.get('training', None)

    if training_flag is not None:
        train_params = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer_dict,
            'scheduler_dict': scheduler_dict,
            "device_dict": device_dict,
            'train_loaders': train_loaders,
            'test_loaders': test_loaders,
            'logger': logger,
            'logdir': logdir,
        }
        # process yml train params
        yml_training_params = cfg['training']

        pretrained_model = cfg['training'].get('pretrained_model', None)
        if pretrained_model is not None:
            yml_training_params.pop('pretrained_model')
        checkpoint_file = cfg['training'].get('checkpoint', None)
        if checkpoint_file is not None:
            yml_training_params.pop('checkpoint')

        train_params.update(cfg['training'])
        # 针对debug模式，修改log_interval和val_interval
        if debug_flag:
            train_params['log_interval'] = args.debug_log_interval
            train_params['val_interval'] = args.debug_val_interval

        trainer = trainer_class(cuda=cuda_flag, **train_params)

        # 加载预训练模型
        if pretrained_model is not None:
            if '~' in pretrained_model:
                pretrained_model = os.path.expanduser(pretrained_model)
            # assert os.path.isfile(pretrained_model), '{} is not a weight file'.format(pretrained_model)
            logger.info('Load pretrained model in {}'.format(pretrained_model))
            trainer.load_pretrained_model(pretrained_model)

        # 恢复训练
        if checkpoint_file is not None:
            if '~' in checkpoint_file:
                checkpoint_file = os.path.expanduser(checkpoint_file)
            trainer.resume_training(checkpoint_file)

        trainer()

    else:
        assert cfg['testing'] is not None, 'you should specify training or testing mode'
        test_params = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer_dict,
            'scheduler_dict': scheduler_dict,
            "device_dict": device_dict,
            'train_loaders': train_loaders,
            'test_loaders': test_loaders,
            'logger': logger,
            'logdir': logdir,
        }

        if cfg['testing'].get('checkpoint', None) is not None:
            tested_model_path = cfg['testing']['checkpoint']
            if '~' in tested_model_path:
                tested_model_path = os.path.expanduser(tested_model_path)
            cfg['testing'].pop('checkpoint')
            load_type = 1
        elif cfg['testing'].get('pretrained_model', None) is not None:
            tested_model_path = cfg['testing']['pretrained_model']
            if '~' in tested_model_path:
                tested_model_path = os.path.expanduser(tested_model_path)
            cfg['testing'].pop('pretrained_model')
            load_type = 2
        else:
            raise RuntimeError('weights or pretrained model need specify')
        test_params.update(cfg['testing'])
        trainer = trainer_class(cuda=cuda_flag, **test_params)

        # 加载模型参数
        if load_type == 1:
            trainer.resume_training(tested_model_path)
        else:
            trainer.load_pretrained_model(tested_model_path)
        # 测试
        trainer.validator(iteration=0)


if __name__ == "__main__":
    project_root = os.path.expanduser('~/PycharmProjects/CGAN_DA')
    data_root = os.path.expanduser('~/PycharmProjects/CGAN_DA/data')
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--job_id', default='debug')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--train_debug_sample_num', type=int, default=10)
    parser.add_argument('--test_debug_sample_num', type=int, default=10)
    parser.add_argument('--debug_log_interval', type=int, default=1)
    parser.add_argument('--debug_val_interval', type=int, default=8)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--trainer', help='trainer classes', default='multiviewadv')
    # parser.add_argument('--trainer', help='trainer classes', default='adaindet')
    parser.add_argument('--data_root', help='dataset root path', default=data_root)
    parser.add_argument('--task_type', help='segmentation or detection', default="cls")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        # default=project_root + "/configs/detection_da_config/cluster_align_online_pseudo/cluster_align_online_det_pseudo_label_tgt_label_guided_margin_0.8_balanced_sample_min_10_pure_online_group_1_test.yml",
        # default=project_root + '/configs/detection_da_config/cluster_align_center_loss/cluster_align_online_det_tgt_plabel_guided_margin_0.8_bamin_10_g1_center_margin_0_pure_online_test.yml',
        # default=project_root+'/configs/detection_da_config/ada_margin/ada_margin_ba10_center_m1_from_0_fix_label_0_test.yml',
        default=project_root + "/configs/multiview_adv/multiview_adv_office_home_A_C_sample_cls_24_use_transformer_test.yml",
        # default=project_root + "/configs/detection_da_config/rpn_simplify_kitti_5c/rpn_simplify_kitti_5c_view_mask.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()
    debug_flag = args.debug
    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4],
                          'job_' + args.job_id + '_exp_' + str(run_id))
    print('logdir is {}'.format(logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)  #
    shutil.copytree('detda', os.path.join(logdir, 'source_code'))  # 拷贝代码
    shutil.copy('./training_scripts.py', os.path.join(logdir, 'source_code'))

    new_config_file = os.path.join(logdir, os.path.basename(args.config))
    with open(new_config_file) as fp:
        # cfg = yaml.load(fp,Loader=yaml.FullLoader)
        cfg = yaml.load(fp)
    # 检测任务依赖于全局的cfg，其中一些需要在模型初始化的时候就用到
    if args.task_type == 'det':
        from detda.models.det_models.utils.config import cfg_from_dict

        cfg_from_dict(cfg['config_dict'])
    logger = get_logger(logdir)
    logger.info('Let the games begin')
    logger.info('Job ID in Cluster is {}'.format(args.job_id))

    train(cfg, logger, logdir, args)