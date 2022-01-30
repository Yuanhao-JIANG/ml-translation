import config
from fairseq import utils
from fairseq.tasks.translation import TranslationTask


def config_task():
    task_config = config.get_translation_task_config()
    task = TranslationTask.setup_task(task_config)
    return task


def load_dataset(task):
    task.load_dataset(split="train", epoch=1, combine=True)  # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True, seed=1):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator
