import os
import torch
import argparse
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
task_args = task_parser.parse_args([config.data_dir])
task = TranslationTask.setup_task(task_args)

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
    with open("./temp/test.raw.en", "w")as f:
        f.write(src)
    with open("./temp/test.raw.zh", "w")as f:
        f.write(src)

    data_handle.clean_corpus("./temp/test.raw", "en", "zh", ratio=-1, min_len=-1, max_len=-1, allow_reclean=True)

    for lang in ["en", "zh"]:
        out_path, in_path = Path(f'./temp/test.{lang}'), Path(f'./temp/test.raw.clean.{lang}')
        with open(out_path, 'w') as out_f:
            with open(in_path, 'r') as in_f:
                for line in in_f:
                    line = line.strip()
                    tok = spm_model.encode(line, out_type=str)
                    print(' '.join(tok), file=out_f)

    # TODO: try --only-source argument in the following cmd
    cmd = "python -m fairseq_cli.preprocess --source-lang en --target-lang zh " \
          "--srcdict " + config.data_dir + "/dict.en.txt --tgtdict " + config.data_dir + "/dict.zh.txt " \
          "--testpref ./temp/test --destdir ./temp --workers 2"
    os.system(cmd)

    # TODO: load data and translate
