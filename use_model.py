import os
import torch
import shutil
import logging
import argparse
import data_util
import data_handle
from pathlib import Path
from fairseq import utils
import sentencepiece as spm
from argparse import Namespace
from config import get_model_architecture_config
from fairseq.tasks.translation import TranslationTask
from model_architecture.seq2seq_model import build_model


# logging.disable()

# config
config = Namespace(
    data_dir="./data/data-bin",
    model_path="./saved_model/RNN/checkpoint_best.pt",
    model_type="RNN",
    spm_model_path="./data/rawdata/spm8000.model",

    # beam size for beam search
    beam=5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process="sentencepiece",
)


def process(src, from_file=False):
    # write and clean data
    if not from_file:
        with open("./temp/test.raw.en", "w") as f:
            f.write(src)
        with open("./temp/test.raw.zh", "w") as f:
            f.write(src)
    else:
        shutil.copyfile(src, "./temp/test.raw.en")
        shutil.copyfile(src, "./temp/test.raw.zh")
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
    cmd = "python -m fairseq_cli.preprocess --source-lang en --target-lang zh " \
          "--srcdict " + config.data_dir + "/dict.en.txt --tgtdict " + config.data_dir + "/dict.zh.txt " \
          "--testpref ./temp/test --destdir ./temp --workers 2"
    os.system(cmd)


model_path = Path(config.model_path)
spm_model_path = Path(config.spm_model_path)
if not model_path.exists():
    print(f"{model_path} doesn't exist!")
    exit()
if not spm_model_path.exists():
    print(f"{spm_model_path} doesn't exist!")

# set spm model
spm_model = spm.SentencePieceProcessor(model_file=str(spm_model_path))

# preprocess
process("a")

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set up task
task_parser = argparse.ArgumentParser()
TranslationTask.add_args(task_parser)
task_args = task_parser.parse_args(["./temp"])
additional_task_args = Namespace(
    dataset_impl="mmap",
    required_seq_len_multiple=8,
)
task_args = Namespace(**vars(task_args), **vars(additional_task_args))
task = TranslationTask.setup_task(task_args)

# set up model
model_arch_args = get_model_architecture_config(config.model_type)
model = build_model(model_arch_args, task, model_type=config.model_type)
model = model.to(device=device)
check = torch.load(model_path)
model.load_state_dict(check["model"])
stats = check["stats"]
step = check["optim"]["step"]
print(f"loaded model {model_path}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
model.eval()

# fairseq's beam search generator
# given model and input sequence, produce translation hypotheses by beam search
sequence_generator = task.build_generator([model], config)


def decode(tokens, dictionary):
    # convert from Tensor to human-readable sentence
    s = dictionary.string(
        tokens.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"


def inference_step(sample):
    srcs, hyps = [], []
    gen_out = sequence_generator.generate([model], sample)
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"],  # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
    return srcs, hyps


def translate(src, output_path="./output.txt", from_file=False, output_to_file=False):
    """
    :param src: String to be translated if from_file=False;
                String of the path of the file to be translated if from_file=True, relative to the root directory,
                e.g. "./testfile.txt"
    :param output_path: String of the path of the file which the output to be written to if output_to_file if True,
                        relative to the root directory, e.g. "./testoutput.txt"
    :param from_file: If True, translate from the given file, otherwise translated the given sentence
    :param output_to_file: If True, write translate output to the given file, otherwise print the output
    :return: None (just print the translated result)
    """
    process(src, from_file)

    # load and translate data
    # global task
    global sequence_generator
    if sequence_generator is None:
        sequence_generator = task.build_generator([model], config)
    task.load_dataset(split="test", epoch=1)
    itr = data_util.load_data_iterator(task, "test", 1, 8192, 2).next_epoch_itr(shuffle=False)
    hyps = []
    with torch.no_grad():
        for i, sample in enumerate(itr):
            sample = utils.move_to_cuda(sample, device=device)
            s, h = inference_step(sample)
            hyps.extend(h)
            if not output_to_file:
                print(hyps)
            else:
                with open(output_path, "w") as f:
                    for line in hyps:
                        f.write(line+"\n")
