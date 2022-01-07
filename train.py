import config
import data_util
import data_handle


# seed = 1
# config.set_seed(seed)


def main():
    data_handle.main()

    logger = config.get_logger()
    task = data_util.config_task()
    data_util.load_dataset(task)


if __name__ == '__main__':
    main()
