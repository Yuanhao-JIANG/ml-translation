import sys
import logging
from pathlib import Path
import random
import torch
import numpy as np
import argparse
from argparse import Namespace
from fairseq.tasks.translation import TranslationTask


def get_general_config():
    config = Namespace(
        datadir="./data/data-bin",
        savedir="./checkpoints/rnn",
        source_lang="en",
        target_lang="zh",

        # cpu threads when fetching & processing data.
        num_workers=2,
        # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
        max_tokens=8192,
        accum_steps=2,

        # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
        lr_factor=2.,
        lr_warmup=4000,

        # clipping gradient norm helps alleviate gradient exploding
        clip_norm=1.0,

        # maximum epochs for training
        max_epoch=30,
        start_epoch=1,

        # beam size for beam search
        beam=5,
        # generate sequences of maximum length ax + b, where x is the source length
        max_len_a=1.2,
        max_len_b=10,
        # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
        post_process="sentencepiece",

        # checkpoints
        keep_last_epochs=5,
        resume=None,  # if resume from checkpoint name (under config.savedir)

        # logging
        use_wandb=False,
    )
    return config


def get_translation_task_config():
    config = get_general_config()

    parser = argparse.ArgumentParser()
    TranslationTask.add_args(parser)
    args = parser.parse_args([config.datadir, '--source-lang', config.source_lang, '--target-lang', config.target_lang,
                              '--upsample-primary', '1'])

    additional_args = Namespace(
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
    )

    task_config = Namespace(**vars(args), **vars(additional_args))

    return task_config


def get_model_architecture_config(model_type="RNN"):
    if model_type != "RNN" and model_type != "Transformer":
        raise Exception("Unknown model" + model_type)

    model_arch_args = Namespace(
        encoder_embed_dim=256,
        encoder_ffn_embed_dim=512,
        encoder_layers=1,
        decoder_embed_dim=256,
        decoder_ffn_embed_dim=1024,
        decoder_layers=1,
        share_decoder_input_output_embed=True,
        dropout=0.3,
    )

    def add_transformer_args(args):
        args.encoder_attention_heads = 4
        args.encoder_normalize_before = True

        args.decoder_attention_heads = 4
        args.decoder_normalize_before = True

        args.activation_fn = "relu"
        args.max_source_positions = 1024
        args.max_target_positions = 1024

        # patches on default parameters for Transformer (those not set above)
        from fairseq.models.transformer import base_architecture
        base_architecture(args)

    if model_type == "Transformer":
        add_transformer_args(model_arch_args)

    if get_general_config().use_wandb:
        import wandb
        wandb.config.update(vars(model_arch_args))

    return model_arch_args


def get_logger():
    config = get_general_config()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "ml_translation"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)
    return logger


def set_seed(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
