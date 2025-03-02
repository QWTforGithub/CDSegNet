"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():

    dataset = "scannet" # {scannet, scannet200, nuscenes}
    config = "CDSegNet" # {CDSegNet, PTv3_CNF}

    num_gpus = 2
    config_file = f"../configs/{dataset}/{config}.py"

    # the path of saving results
    options = {'save_path': f'../exp/{dataset}/{config}'}

    args = default_argument_parser().parse_args()
    args.config_file = config_file
    args.num_gpus = num_gpus
    args.options = options

    cfg = default_config_parser(args.config_file, args.options)

    # the number of GPUs
    cfg.num_gpus = num_gpus

    # checkpoint path

    weight = f"../exp/{dataset}/{config}/model/model_last.pth"
    cfg.weight = weight
    cfg.resume = True

    # After {save_freq_threshold} epochs, the checkpoint is saved every {save_freq} epochs.
    save_freq = 1
    save_freq_threshold = 70
    cfg.save_freq = save_freq
    cfg.hooks[4].save_freq = save_freq
    cfg.save_freq_threshold = save_freq_threshold

    if(cfg.data_root.__contains__("scannet_debug")):
        cfg.eval_epoch = cfg.epoch = 1
        cfg.data.train.loop = 1

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    main()
