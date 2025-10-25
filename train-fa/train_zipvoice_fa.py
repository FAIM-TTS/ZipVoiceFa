#!/usr/bin/env python3
# Copyright  2024-2025  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train ZipVoice with flow-matching loss using Hugging Face Accelerate.

Launch example (FP16 handled by Accelerate):

accelerate launch --num_processes 8 --multi_gpu -m zipvoice.bin.train_zipvoice \
  --num-epochs 11 \
  --max-duration 500 \
  --lr-hours 30000 \
  --model-config conf/zipvoice_base.json \
  --tokenizer emilia \
  --token-file data/tokens_emilia.txt \
  --dataset emilia \
  --manifest-dir data/fbank \
  --exp-dir exp/zipvoice
"""

import argparse
import copy
import json
import logging
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tqdm.auto import tqdm


import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed as accel_set_seed


from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from zipvoice.utils.common import (
    AttributeDict,
    MetricsTracker,
    get_adjusted_batch_count,
    get_env_info,
    get_parameter_groups_with_lrs,
    prepare_input,
    set_batch_count,
    setup_logger,
    str2bool,
)
from zipvoice.utils.lr_scheduler import Eden, FixedLRScheduler, LRScheduler
from zipvoice.utils.optim import ScaledAdam

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Log to TensorBoard (main process only).",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=11,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=0,
        help="Number of iterations to train; ignores num-epochs if > 0.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Resume from this epoch (>1 loads exp-dir/epoch-{start_epoch-1}.pt).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained checkpoint to initialize the model.",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice",
        help="Directory for checkpoints, logs, etc.",
    )

    parser.add_argument("--base-lr", type=float, default=0.02, help="Base LR.")

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="LR schedule parameter (keep default).",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="LR schedule parameter (epochs).",
    )

    parser.add_argument(
        "--lr-hours",
        type=float,
        default=0,
        help="If > 0, use hours-based LR schedule instead of epochs.",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=50,
        help="Reference batch duration for schedule adjustment.",
    )

    parser.add_argument(
        "--finetune",
        type=str2bool,
        default=False,
        help="Use fixed LR schedule and skip large dropout phase.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5000,
        help="Save checkpoint every N training batches.",
    )

    parser.add_argument(
        "--valid-by-epoch",
        type=str2bool,
        default=False,
        help="If True, validate once per epoch (at batch 0). Otherwise validate every save interval.",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="Keep only the last K step-checkpoints (epoch checkpoints unaffected).",
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="Update running averaged model every N batches (main process only).",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="Scale factor for fbank features.",
    )

    parser.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=0.2,
        help="Drop rate of text condition during training.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "custom"],
        help="Training dataset choice.",
    )

    parser.add_argument("--train-manifest", type=str, help="Path to train manifest")
    parser.add_argument("--dev-manifest", type=str, help="Path to dev/val manifest")

    parser.add_argument(
        "--min-len", type=float, default=1.0, help="Min audio length (s)"
    )
    parser.add_argument(
        "--max-len", type=float, default=30.0, help="Max audio length (s)"
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_base.json",
        help="Model configuration JSON.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language id for espeak tokenizer.",
    )

    parser.add_argument(
        "--token-file",
        type=str,
        default="data/tokens_emilia.txt",
        help="Token-id mapping file: '{token}\\t{token_id}' per line.",
    )

    # Add dataset module args
    TtsDataModule.add_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "env_info": get_env_info(),
        }
    )
    return params


def compute_fbank_loss(
    params: AttributeDict,
    model: nn.Module,
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    device = next(model.parameters()).device

    batch_size, _, _ = features.shape
    noise = torch.randn_like(features)

    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (torch.arange(batch_size, device=device) / batch_size).unsqueeze(1).unsqueeze(2)

    with torch.set_grad_enabled(is_training):
        loss = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            condition_drop_ratio=params.condition_drop_ratio,
        )

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    return loss, info


def train_one_epoch(
    accelerator: Accelerator,
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> None:
    is_main = accelerator.is_main_process
    model.train()
    tot_loss = MetricsTracker()

    total_steps = None
    try:
        total_steps = len(train_dl)
    except Exception:
        try:
            total_steps = len(getattr(train_dl, "sampler", None))
        except Exception:
            total_steps = None

    pbar = tqdm(
        total=total_steps,
        disable=not is_main,
        dynamic_ncols=True,
        leave=False,
        desc=f"train e{params.cur_epoch}",
    )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(
                model,
                get_adjusted_batch_count(params) + (100000 if params.finetune else 0),
            )

        # validation cadence
        if (params.valid_by_epoch and batch_idx == 0) or (
            not params.valid_by_epoch and params.batch_idx_train % params.valid_interval == 0
        ):
            if is_main:
                logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                accelerator=accelerator, params=params, model=model, valid_dl=valid_dl
            )
            model.train()
            if is_main:
                logging.info(
                    f"Epoch {params.cur_epoch}, global_batch_idx: {params.batch_idx_train}, validation: {valid_info}"
                )
                if tb_writer is not None:
                    valid_info.write_summary(tb_writer, "train/valid_", params.batch_idx_train)

        params.batch_idx_train += 1

        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=accelerator.device,
            return_tokens=True,
            return_feature=True,
        )

        with accelerator.autocast():
            loss, loss_info = compute_fbank_loss(
                params=params,
                model=model,
                features=features,
                features_lens=features_lens,
                tokens=tokens,
                is_training=True,
            )

        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        accelerator.backward(loss)
        scheduler.step_batch(params.batch_idx_train)
        if params.lr_hours > 0:
            scheduler.step_epoch(
                params.batch_idx_train * params.max_duration * accelerator.num_processes / 3600
            )
        optimizer.step()
        optimizer.zero_grad()

        # averaged model (main only)
        if is_main and params.batch_idx_train > 0 and (params.batch_idx_train % params.average_period == 0):
            update_averaged_model(
                params=params,
                model_cur=accelerator.unwrap_model(model),
                model_avg=model_avg,
            )

        # periodic checkpoint (main only)
        if is_main and params.batch_idx_train > 0 and params.batch_idx_train % params.save_every_n == 0:
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=accelerator.unwrap_model(model),
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=None,
                rank=0,
            )
            remove_checkpoints(out_dir=params.exp_dir, topk=params.keep_last_k, rank=0)

        # tqdm update (show current loss + LR)
        if is_main:
            try:
                cur_lr = max(scheduler.get_last_lr())
            except Exception:
                cur_lr = float("nan")
            pbar.update(1)
            cur_lpf = loss_info["loss"] / max(1, loss_info["frames"])  # per-frame
            ema_lpf = tot_loss["loss"] / max(1, tot_loss["frames"])    # EMA per-frame
            pbar.set_postfix_str(f"lpf={cur_lpf:.4f} ema={ema_lpf:.4f}")

            if tb_writer is not None and (params.batch_idx_train % params.log_interval == 0):
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
                loss_info.write_summary(tb_writer, "train/current_", params.batch_idx_train)
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

    if is_main:
        pbar.close()

    params.train_loss = tot_loss["loss"]
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss



def compute_validation_loss(
    accelerator: Accelerator,
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
) -> MetricsTracker:
    model.eval()

    tot_loss = MetricsTracker()

    with torch.no_grad():
        for _, batch in enumerate(valid_dl):
            tokens, features, features_lens = prepare_input(
                params=params,
                batch=batch,
                device=accelerator.device,
                return_tokens=True,
                return_feature=True,
            )

            loss, loss_info = compute_fbank_loss(
                params=params,
                model=model,
                features=features,
                features_lens=features_lens,
                tokens=tokens,
                is_training=False,
            )
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info

    if accelerator.num_processes > 1:
        tot_loss.reduce(next(model.parameters()).device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def tokenize_text(c: Cut, tokenizer):
    if hasattr(c.supervisions[0], "tokens"):
        tokens = tokenizer.tokens_to_token_ids([c.supervisions[0].tokens])
    else:
        tokens = tokenizer.texts_to_token_ids([c.supervisions[0].text])
    c.supervisions[0].tokens = tokens[0]
    return c


def run(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # seed *all* RNGs
    accel_set_seed(args.seed, device_specific=True) 

    is_main = accelerator.is_main_process

    params = get_params()
    params.update(vars(args))
    params.valid_interval = params.save_every_n

    params.world_size = max(1, accelerator.num_processes)
    params.rank = accelerator.process_index
    params.device = accelerator.device

    if params.num_iters > 0:
        params.num_epochs = 1_000_000

    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])


    fix_random_seed(params.seed)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard") if (args.tensorboard and is_main) else None

    # Tokenizer
    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=params.token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=params.token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=params.token_file, lang=params.lang)
    else:
        tokenizer = SimpleTokenizer(token_file=params.token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    params.update(tokenizer_config)

    if is_main:
        logging.info(params)
        logging.info("Creating model")

    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )

    if params.checkpoint is not None:
        if is_main:
            logging.info(f"Loading pre-trained model from {params.checkpoint}")
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=True)

    num_param = sum(p.numel() for p in model.parameters())
    if is_main:
        logging.info(f"Number of parameters : {num_param}")

    # running average model (main only)
    model_avg: Optional[nn.Module] = copy.deepcopy(model).to(torch.float64) if is_main else None

    # resume (epoch-1) if requested
    if params.start_epoch > 1:
        checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)
    else:
        checkpoints = None

    # optimizer & sched
    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,
        clipping_scale=2.0,
    )

    if params.finetune:
        scheduler = FixedLRScheduler(optimizer)
    elif params.lr_hours > 0:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_hours)
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    # restore optimizer/scheduler if resuming
    if params.start_epoch > 1 and checkpoints is not None:
        if "optimizer" in checkpoints:
            if is_main:
                logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])
        if "scheduler" in checkpoints:
            if is_main:
                logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    # Data
    def _remove_short_and_long_utt(c: Cut, min_len: float, max_len: float):
        return (min_len <= c.duration <= max_len)

    datamodule = TtsDataModule(args)

    if params.dataset == "emilia":
        train_cuts = CutSet.mux(
            datamodule.train_emilia_EN_cuts(),
            datamodule.train_emilia_ZH_cuts(),
            weights=[46000, 49000],
        ).filter(partial(_remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len))

        dev_cuts = CutSet.mux(
            datamodule.dev_emilia_EN_cuts(),
            datamodule.dev_emilia_ZH_cuts(),
            weights=[0.5, 0.5],
        )
    elif params.dataset == "libritts":
        train_cuts = datamodule.train_libritts_cuts().filter(
            partial(_remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len)
        )
        dev_cuts = datamodule.dev_libritts_cuts()
    else:
        # custom
        train_cuts = datamodule.train_custom_cuts(params.train_manifest).filter(
            partial(_remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len)
        )
        dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest).filter(
            partial(_remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len)
        )

    # tokenization map
    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)
    train_cuts = train_cuts.map(_tokenize_text)
    dev_cuts = dev_cuts.map(_tokenize_text)

    train_dl = datamodule.train_dataloaders(train_cuts)
    valid_dl = datamodule.dev_dataloaders(dev_cuts)
    model, optimizer = accelerator.prepare(model, optimizer)

    if is_main:
        logging.info("Training started")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if params.lr_hours == 0:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)

        # epoch to sampler for shuffling in distributed
        if hasattr(train_dl, "sampler") and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            accelerator=accelerator,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
        )

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

        if accelerator.is_main_process:
            filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=accelerator.unwrap_model(model),
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=None,  # unused
                rank=0,
            )

            if params.best_train_epoch == params.cur_epoch:
                copyfile(src=filename, dst=params.exp_dir / "best-train-loss.pt")
            if params.best_valid_epoch == params.cur_epoch:
                copyfile(src=filename, dst=params.exp_dir / "best-valid-loss.pt")

    if is_main:
        logging.info("Done!")

    accelerator.wait_for_everyone()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run(args)


if __name__ == "__main__":
    main()
