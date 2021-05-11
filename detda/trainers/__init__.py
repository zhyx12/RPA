# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from detda.trainers.trainer_basic_detection import TrainerBasicDetection
from detda.trainers.trainer_basic_adv_detection import TrainerBasicAdvDetection
from detda.trainers.trainer_rpn_cluster_align import TrainerRPNClusterAlign


def get_trainer(name):
    return _get_trainer_instance(name)


def _get_trainer_instance(name):
    try:
        return {
            "basicdetection": TrainerBasicDetection,
            "basicadvdetection": TrainerBasicAdvDetection,
            "rpnclusteralign": TrainerRPNClusterAlign,
        }[name]
    except:
        raise RuntimeError("Trainer {} not available".format(name))
