from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

import yaml
import math
import mlflow
import mlflow.pytorch

from data_processing.make_dataset_function import make_training_dataset


def main(opt):
    make_training_dataset(
        json_path="../data/merged_dataset/person_keypoints.json",
        all_images_dir="../data/merged_dataset/",
        train_dir="../data/active/train/",
        val_dir="../data/active/val/",
        train_json_out="../data/active/annotations/active_train.json",
        val_json_out="../data/active/annotations/active_val.json",
        val_ratio=0.12,
        general_val_dir="../data/general_val_500/",
        use_existing_train_set=True,
        use_existing_val=True,
        augmentations=None
    )

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.froze_backbone)
    
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")

    # Only include parameters that are NOT in the backbone
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]

    optimizer = torch.optim.AdamW(
        [
            {'params': head_params, 'lr': opt.lr, 'name': 'head'},
            {'params': backbone_params, 'lr': opt.backbone_lr, 'name': 'backbone_partial'},
        ],
        weight_decay=opt.weight_decay,
        betas=(opt.beta1, opt.beta2),
        eps=opt.epsilon
    )

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Scheduler setup
    steps_per_epoch = len(train_loader)
    total_steps = opt.num_epochs * steps_per_epoch

    warmup_steps = int(opt.warmup_epochs * steps_per_epoch)
    unfreeze_warmup_steps = int(opt.unfreeze_warmup_epochs * steps_per_epoch)

    backbone_unfreeze_step = None

    # Keep track of names
    param_group_names = [g['name'] for g in optimizer.param_groups]

    def make_lambda(name):
        def lr_lambda(step):
            if 'backbone' in name:
                if backbone_unfreeze_step is not None:
                    backbone_step = step - backbone_unfreeze_step
                    if backbone_step < unfreeze_warmup_steps:
                        return backbone_step / max(1, unfreeze_warmup_steps)
                    else:
                        progress = (backbone_step - unfreeze_warmup_steps) / max(
                            1, total_steps - backbone_unfreeze_step - unfreeze_warmup_steps
                        )
                        return 0.5 * (1 + math.cos(progress * math.pi))
                else:
                    return 0.0
            else:  # head
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                else:
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(progress * math.pi))
        return lr_lambda

    lambdas = [make_lambda(name) for name in param_group_names]
    scheduler = LambdaLR(optimizer, lr_lambda=lambdas)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer, scheduler)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # MLflow setup
    mlflow.set_tracking_uri("http://35.165.139.156:5000")
    mlflow.set_experiment(opt.mlflow_experiment)
    with mlflow.start_run(run_name=f"{opt.exp_id}"):
        mlflow.log_params({
            "architecture": "MoveNet Thunder",
            "batch_size": opt.batch_size,
            "epochs": opt.num_epochs,
            "datasets": opt.datasets,
            "optimizer": "AdamW",
            "learning_rate": opt.lr,
            "backbone_lr": opt.backbone_lr,
            "lr_schedule": "cosine_decay",
            "weight_decay": opt.weight_decay,
            "beta1": opt.beta1,
            "beta2": opt.beta2,
            "epsilon": opt.epsilon,
            "warmup_epochs": opt.warmup_epochs,
            "unfreeze_warmup_epochs": opt.unfreeze_warmup_epochs,
            "unfreeze_backbone_epoch": opt.unfreeze_epoch,
            "start_unfreeze_layer": opt.start_unfreeze,
            "end_unfreeze_layer": opt.end_unfreeze,
            "unfreeze_fpn": opt.unfreeze_fpn,
            "gradient_clip_max_norm": opt.max_norm,
            "save_metric": opt.metric,
            "val_intervals": opt.val_intervals,
        })
        mlflow.set_tag("datasets", " + ".join(opt.datasets))

        print('Starting training...')
        patience = 10 / opt.val_intervals
        counter = 0
        best = 0 # Make this 1e10 when using loss

        for epoch in range(start_epoch + 1, opt.num_epochs + 1):
            # Partial unfreeze at unfreeze_epoch
            if epoch == opt.unfreeze_epoch:
                print(f"Partially unfreezing backbone parameters at epoch {epoch}")
                
                for name, param in model.backbone.named_parameters():
                    body_match = any(f"body.{i}" in name for i in range(opt.start_unfreeze, opt.end_unfreeze + 1))
                    fpn_match = "fpn" in name if opt.unfreeze_fpn else False

                    if body_match or fpn_match:
                        param.requires_grad = True

                backbone_unfreeze_step = trainer.global_step
                scheduler.step()  # immediately apply warmup

            # Train epoch
            mark = epoch if opt.save_all else 'last'
            log_dict_train, _ = trainer.train(epoch, train_loader)

            # Logging
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
                mlflow.log_metric(f"train/{k}", v, step=epoch)

            # Validation
            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                model_path = os.path.join(opt.save_dir, f'model_{mark}.pth')
                save_model(model_path, epoch, model, optimizer)
                with torch.no_grad():
                    log_dict_val, preds = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    logger.write('{} {:8f} | '.format(k, v))
                    mlflow.log_metric(f"val/{k}", v, step=epoch)
                if log_dict_val[opt.metric] > best: # Make this < when using loss rather than mAP
                    best = log_dict_val[opt.metric]
                    print(f"Best {opt.metric} at epoch {epoch}: {best}")
                    best_model_path = os.path.join(opt.save_dir, 'model_best.pth')
                    save_model(best_model_path, epoch, model)
                else:
                    counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}, no improvement for {patience} epochs.")
                    break
            else:
                model_path = os.path.join(opt.save_dir, 'model_last.pth')
                save_model(model_path, epoch, model, optimizer)
            logger.write('\n')

        # Final logging & artifact save
        final_last_model_path = os.path.join(opt.save_dir, 'model_last.pth')
        final_best_model_path = os.path.join(opt.save_dir, 'model_best.pth')
        mlflow.log_artifact(final_last_model_path)
        mlflow.log_artifact(final_best_model_path)
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.log_artifact("../data/active/annotations/active_train.json", artifact_path="annotations")
        mlflow.log_artifact("../data/active/annotations/active_val.json", artifact_path="annotations")
        mlflow.log_artifact("data_processing/augmentations.yaml", artifact_path="annotations")

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    with open(opt.hyperparam_yaml) as f:
        sweep_config = yaml.safe_load(f)["sweep_runs"]
    for run_cfg in sweep_config:
        opt.exp_id = f"HEEL_SLIDES_{run_cfg['name']}"
        opt.lr = float(run_cfg["head_lr"])
        opt.backbone_lr = float(run_cfg["backbone_lr"])
        opt.unfreeze_epoch = int(run_cfg["unfreeze_epoch"])
        opt.warmup_epochs = float(run_cfg["warmup_epochs"])
        opt.unfreeze_warmup_epochs = float(run_cfg["unfreeze_warmup_epochs"])
        opt.weight_decay = float(run_cfg["weight_decay"])
        opt.batch_size = int(run_cfg["batch_size"])
        opt.num_epochs = int(run_cfg["num_epochs"])
        opt.start_unfreeze = int(run_cfg["start_unfreeze"])
        opt.end_unfreeze = int(run_cfg["end_unfreeze"])
        opt.unfreeze_fpn = bool(run_cfg["unfreeze_fpn"])

        main(opt)