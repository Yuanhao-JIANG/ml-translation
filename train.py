import torch
import config
import data_util
import data_handle
from model_architecture.seq2seq_model import build_model
from model_architecture.optimization import LabelSmoothedCrossEntropyCriterion, NoamOpt


# seed = 1
# config.set_seed(seed)


def main():
    data_handle.main()

    general_config = config.get_general_config()

    logger = config.get_logger()
    task = data_util.config_task()
    data_util.load_dataset(task)

    model_arch_args = config.get_model_architecture_config()
    model = build_model(model_arch_args, task)
    logger.info(model)

    criterion = LabelSmoothedCrossEntropyCriterion(smoothing=0.1, ignore_index=task.target_dictionary.pad())
    optimizer = NoamOpt(model_size=model_arch_args.encoder_embed_dim, factor=general_config.lr_factor,
                        warmup=general_config.lr_warmup,
                        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,
                                                    weight_decay=0.0001))


if __name__ == '__main__':
    main()
