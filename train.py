import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

import config
import data_util
import valid_test
import data_handle
from model_architecture.seq2seq_model import build_model
from model_architecture.optimization import LabelSmoothedCrossEntropyCriterion, NoamOpt


def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)  # gradient accumulation: update every accum_steps samples

    stats = {"loss": []}
    scaler = GradScaler()  # automatic mixed precision (amp)

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))  # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(),
                                         general_config.clip_norm)  # grad norm clipping prevents gradient exploding

        scaler.step(optimizer)
        scaler.update()

        # logging
        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if general_config.use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats


# seed = 1
# config.set_seed(seed)
# data_handle.main()
logger = config.get_logger()
general_config = config.get_general_config()
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    task = data_util.config_task()
    data_util.load_dataset(task)
    epoch_loader = data_util.load_data_iterator(task, "train", general_config.start_epoch, general_config.max_tokens,
                                                general_config.num_workers)

    model_arch_args = config.get_model_architecture_config()
    model = build_model(model_arch_args, task)
    logger.info(model)

    criterion = LabelSmoothedCrossEntropyCriterion(smoothing=0.1, ignore_index=task.target_dictionary.pad())
    optimizer = NoamOpt(model_size=model_arch_args.encoder_embed_dim, factor=general_config.lr_factor,
                        warmup=general_config.lr_warmup,
                        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,
                                                    weight_decay=0.0001))

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {general_config.max_tokens}, accumulate steps = {general_config.accum_steps}")

    # train and validation and save
    sequence_generator = valid_test.generate_sequence_generator(task, model)
    valid_test.try_load_checkpoint(model, logger, optimizer, name=general_config.resume)
    while epoch_loader.next_epoch_idx <= general_config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_loader, model, task, criterion, optimizer, general_config.accum_steps)
        stats = valid_test.validate_and_save(model, task, criterion, optimizer, epoch_loader.epoch, logger,
                                             sequence_generator)
        logger.info("end of epoch {}".format(epoch_loader.epoch))
        epoch_loader = data_util.load_data_iterator(task, "train", epoch_loader.next_epoch_idx,
                                                    general_config.max_tokens, general_config.num_workers)

    valid_test.try_load_checkpoint(model, logger, name="checkpoint_best.pt")
    valid_test.validate(model, task, criterion, logger, sequence_generator, log_to_wandb=False)
    generate_prediction(model, task, sequence_generator)


def generate_prediction(model, task, sequence_generator, split="test", outfile="./prediction/prediction.txt"):
    task.load_dataset(split=split, epoch=1)
    itr = data_util.load_data_iterator(task, split, 1, general_config.max_tokens, general_config.num_workers) \
        .next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = valid_test.inference_step(sample, model, task, sequence_generator)

            hyps.extend(h)
            idxs.extend(list(sample['id']))

    # sort based on the order before preprocess
    hyps = [x for _, x in sorted(zip(idxs, hyps))]

    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h + "\n")


if __name__ == '__main__':
    main()
