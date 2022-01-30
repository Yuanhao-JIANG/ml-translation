import os
import torch
import argparse
import data_util
import data_handle
from pathlib import Path
import sentencepiece as spm
from argparse import Namespace
from config import get_model_architecture_config
from model_architecture.seq2seq_model import build_model
from fairseq.tasks.translation import TranslationTask

# config
config = Namespace(
    data_dir="./data/data-bin",
    model_path="./saved_model/RNN/checkpoint_best.pt",
    model_type="RNN",
    spm_model_path="./data/rawdata/spm8000.model",
)

model_path = Path(config.model_path)
spm_model_path = Path(config.spm_model_path)
if not model_path.exists():
    print(f"{model_path} doesn't exist!")
    exit()
if not spm_model_path.exists():
    print(f"{spm_model_path} doesn't exist!")

# set spm model
spm_model = spm.SentencePieceProcessor(model_file=str(spm_model_path))

# set up task
task_parser = argparse.ArgumentParser()
TranslationTask.add_args(task_parser)
task_args = task_parser.parse_args(["./temp"])
additional_task_args = Namespace(
    dataset_impl="mmap",
    required_seq_len_multiple=8,
)
task_args = Namespace(**vars(task_args), **vars(additional_task_args))
task = None

# set up model
model_arch_args = get_model_architecture_config(config.model_type)
model = build_model(model_arch_args, task, model_type=config.model_type)
check = torch.load(model_path)
model.load_state_dict(check["model"])
stats = check["stats"]
step = check["optim"]["step"]
print(f"loaded model {model_path}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
model.eval()


def predict(src):
    # write and clean data
    with open("./temp/test.raw.en", "w")as f:
        f.write(src)
    with open("./temp/test.raw.zh", "w")as f:
        f.write(src)
    data_handle.clean_corpus("./temp/test.raw", "en", "zh", ratio=-1, min_len=-1, max_len=-1, allow_reclean=True)

    # make sub-word data
    for lang in ["en", "zh"]:
        out_path, in_path = Path(f'./temp/test.{lang}'), Path(f'./temp/test.raw.clean.{lang}')
        with open(out_path, 'w') as out_f:
            with open(in_path, 'r') as in_f:
                for line in in_f:
                    line = line.strip()
                    tok = spm_model.encode(line, out_type=str)
                    print(' '.join(tok), file=out_f)

    # data binarization
    # TODO: try --only-source argument in the following cmd
    cmd = "python -m fairseq_cli.preprocess --source-lang en --target-lang zh " \
          "--srcdict " + config.data_dir + "/dict.en.txt --tgtdict " + config.data_dir + "/dict.zh.txt " \
          "--testpref ./temp/test --destdir ./temp --workers 2"
    os.system(cmd)

    # load and translate data
    global task
    if task is None:
        task = TranslationTask.setup_task(task_args)
    task.load_dataset(split="test", epoch=1)
    itr = data_util.load_data_iterator(task, "test", 1, 8192, 2).next_epoch_itr(shuffle=False)
    with torch.no_grad():
        for i, sample in enumerate(itr):
            pass  # TODO: translate
