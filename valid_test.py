import config
import shutil
import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from fairseq import utils
import data_util
import tqdm
from pathlib import Path

general_config = config.get_general_config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search
def generate_sequence_generator(task, model):
    return task.build_generator([model], general_config)


def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        general_config.post_process,
    )
    return s if s else "<unk>"


def inference_step(sample, model, task, sequence_generator):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
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
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


def validate(model, task, criterion, logger, sequence_generator, log_to_wandb=True):
    logger.info('begin validation')
    itr = data_util.load_data_iterator(task, "valid", 1, general_config.max_tokens, general_config.num_workers) \
        .next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)

            # do inference
            s, h, r = inference_step(sample, model, task, sequence_generator)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok = 'zh' if task.args.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs

    if general_config.use_wandb and log_to_wandb:
        import wandb
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)

    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])

    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats


def validate_and_save(model, task, criterion, optimizer, epoch, logger, sequence_generator, save=True):
    stats = validate(model, task, criterion, logger, sequence_generator)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(general_config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        # torch.save(check, savedir / f"checkpoint{epoch}.pt")
        # shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
        # logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        # save epoch samples
        with open(savedir / f"samples{epoch}.{general_config.source_lang}-{general_config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            # print(validate_and_save.best_bleu)
            torch.save(check, savedir / f"checkpoint_best.pt")

        del_file = savedir / f"checkpoint{epoch - general_config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats


def try_load_checkpoint(model, logger, optimizer=None, name=None):
    name = name if name else "checkpoint_best.pt"
    checkpath = Path(general_config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer is not None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")
