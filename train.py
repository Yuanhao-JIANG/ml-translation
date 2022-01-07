import config
import data_util
import data_handle
from model_architecture.seq2seq_model import build_model


# seed = 1
# config.set_seed(seed)


def main():
    data_handle.main()

    logger = config.get_logger()
    task = data_util.config_task()
    data_util.load_dataset(task)

    model = build_model(config.get_model_architecture_config(), task)
    logger.info(model)


if __name__ == '__main__':
    main()
