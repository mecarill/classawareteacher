#!/usr/bin/env python3

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from cat import add_cat_config
from cat.engine.trainer import CATTrainer

# hacky way to register
import cat.data.datasets.builtin
from cat.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from cat.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from cat.modeling.proposal_generator.rpn import PseudoLabRPN
from cat.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from cat.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "cat":
        print(1)
        Trainer = CATTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
