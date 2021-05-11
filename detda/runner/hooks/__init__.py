# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from .hook import Hook
from .priority import get_priority
from .training_hooks import *
from .det_val_metrics import DetValMetrics
from .det_rcnn_confusion_metrics import RCNNConfusionMetrics
from .det_rpn_recall import RPNRecall
from .det_online_pseudo_label import DetOnlinePseudoLabel
from .det_rpn_cluster_align import RPNClusterAlign
from .det_cam_mask import DetCAMMask


def _register_hook(runner, hook, priority="NORMAL"):
    """Register a hook into the hook list.

            Args:
                hook (:obj:`Hook`): The hook to be registered.
                priority (int or str or :obj:`Priority`): Hook priority.
                    Lower value means higher priority.
            """
    assert isinstance(hook, Hook)
    if hasattr(hook, 'priority'):
        raise ValueError('"priority" is a reserved attribute for hooks')
    priority = get_priority(priority)
    hook.priority = priority
    # insert the hook to a sorted list
    inserted = False
    for i in range(len(runner._hooks) - 1, -1, -1):
        if priority >= runner._hooks[i].priority:
            runner._hooks.insert(i + 1, hook)
            inserted = True
            break
    if not inserted:
        runner._hooks.insert(0, hook)


def _build_hook(runner, args, hook_type=None):
    if isinstance(args, Hook):
        return args
    elif isinstance(args, dict):
        assert issubclass(hook_type, Hook)
        return hook_type(**args)
    else:
        raise TypeError('"args" must be either a Hook object'
                        ' or dict, not {}'.format(type(args)))
