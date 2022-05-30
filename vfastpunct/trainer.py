from vfastpunct.arguments import get_train_argument, get_test_argument
from vfastpunct.constants import LOGGER, PUNCCAP_MODEL_MAPPING, PUNC_LABEL2ID, CAP_LABEL2ID
from vfastpunct.datasets import build_dataset, build_punccap_dataset
from vfastpunct.ultis import get_total_model_parameters

from typing import Union, Generator
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler

import os
import glob
import sys
import time
import torch
import itertools


def get_datasets(ddir: Union[str, os.PathLike],
                 tokenizer,
                 dtype: str = 'train',
                 max_seq_length: int = 190,
                 device: str = 'cpu',
                 use_crf: bool = True) -> Generator:
    for fpath in glob.glob(str(Path(ddir + f'/{dtype}_*_splitted.txt'))):
        LOGGER.info(f"Load file {fpath}")
        fname = os.path.basename(fpath)
        yield fname, \
              build_punccap_dataset(fpath, tokenizer, max_seq_length=max_seq_length, device=device, use_crf=use_crf)

def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'pclasses': PUNC_LABEL2ID,
        'cclasses': CAP_LABEL2ID,
        'args': args
    }
    torch.save(saved_data, saved_file)


def validate(model, valid_iterator, scheduler=None, is_test=False):
    start_time = time.time()
    model.eval()
    eval_loss, nb_eval_steps = 0.0, 0.0
    eval_ploss, eval_closs = 0.0, 0.0
    eval_ppreds, eval_plabels = [], []
    eval_cpreds, eval_clabels = [], []
    for idx, batch in tqdm(enumerate(valid_iterator), total=len(valid_iterator), position=0, leave=True):
        loss, ploss, closs, eval_plogits, eval_clogits = model(**batch)
        eval_loss += loss.item()
        eval_ploss += ploss.item()
        eval_closs += closs.item()
        nb_eval_steps += 1
        active_accuracy = batch['label_masks'].view(-1) != 0
        plabels = torch.masked_select(batch['plabels'].view(-1), active_accuracy)
        clabels = torch.masked_select(batch['clabels'].view(-1), active_accuracy)
        eval_plabels.extend(plabels.cpu().tolist())
        eval_clabels.extend(clabels.cpu().tolist())
        if isinstance(eval_plogits[-1], list):
            eval_ppreds.extend(list(itertools.chain(*eval_plogits)))
            eval_cpreds.extend(list(itertools.chain(*eval_clogits)))
        else:
            eval_ppreds.extend(eval_plogits)
            eval_cpreds.extend(eval_clogits)
    epoch_loss = eval_loss / nb_eval_steps
    epoch_ploss = eval_ploss / nb_eval_steps
    epoch_closs = eval_closs / nb_eval_steps
    if is_test:
        preports = classification_report(eval_plabels, eval_ppreds, zero_division=0)
        creports = classification_report(eval_clabels, eval_cpreds, zero_division=0)
        LOGGER.info(f'\tTest Loss: {eval_loss}; Spend time: {time.time() - start_time}')
        LOGGER.info('Punct Report:')
        LOGGER.info(preports)
        LOGGER.info('Cap Report:')
        LOGGER.info(creports)
        return epoch_loss
    else:
        preports = classification_report(eval_plabels, eval_ppreds, output_dict=True, zero_division=0)
        creports = classification_report(eval_clabels, eval_cpreds, output_dict=True, zero_division=0)
        macro_avg = (preports['macro avg']['f1-score'] + creports['macro avg']['f1-score']) / 2
        acc_avg = (preports['accuracy'] + creports['accuracy']) / 2
        if scheduler is not None:
            scheduler.step(macro_avg)
        LOGGER.info(f"{'*' * 10}Validate Summary{'*' * 10}")
        LOGGER.info(f"\tValidation Loss: {eval_loss} (pLoss: {epoch_ploss:.4f}; cLoss: {epoch_closs:.4f});\n"
                    f"\tAccuracy: {acc_avg:.4f} (pAccuracy: {preports['accuracy']:.4f}; cAccuracy: {creports['accuracy']:.4f});\n"
                    f"\tMacro-F1 score: {macro_avg:.4f} (pF1: {preports['macro avg']['f1-score']:.4f}; cF1: {creports['macro avg']['f1-score']:.4f});\n"
                    f"\tSpend time: {time.time() - start_time}")
        return epoch_loss, acc_avg, macro_avg


def train_one_epoch(model, optimizer, train_iterator, max_grad_norm: float = 1.0):
    start_time = time.time()
    tr_loss, nb_tr_steps = 0.0, 0.0
    model.train()
    for idx, batch in tqdm(enumerate(train_iterator), total=len(train_iterator), position=0, leave=True):
        loss, _, _, _, _ = model(**batch)
        tr_loss += loss.item()
        nb_tr_steps += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = tr_loss / nb_tr_steps
    LOGGER.info(
        f"\tTraining Lr: {optimizer.param_groups[0]['lr']}; Loss: {epoch_loss}; Spend time: {time.time() - start_time}")
    return epoch_loss


def test():
    args = get_test_argument()
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    assert os.path.exists(args.model_path), f'`{args.model_path}` not exists! What do you do with the checkpoint file?'
    if device == 'cpu':
        checkpoint_data = torch.load(args.model_path, map_location='cpu')
    else:
        checkpoint_data = torch.load(args.model_path)

    configs = checkpoint_data['args']
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)

    test_dataset = build_dataset(args.data_dir,
                                 tokenizer=tokenizer,
                                 data_type='test',
                                 max_seq_length=configs.max_seq_length,
                                 overwrite_data=args.overwrite_data,
                                 device=device)

    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model_archs = PUNCCAP_MODEL_MAPPING[args.model_arch]
    config = model_archs['config_clss'].from_pretrained(args.model_name_or_path,
                                                        num_plabels=len(PUNC_LABEL2ID),
                                                        num_clabels=len(CAP_LABEL2ID),
                                                        finetuning_task=args.task)
    model = model_archs['model_clss'].from_pretrained(args.model_name_or_path, config=config, from_tf=False)
    model.load_state_dict(checkpoint_data['model'])
    model.to(device)
    validate(model, test_iterator, is_test=True)


def train():
    args = get_train_argument()
    use_crf = True if 'crf' in args.model_arch else False
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found! Where is dataset ??????'

    if not os.path.isdir(args.output_dir):
        LOGGER.info(f"Create saved directory {args.output_dir}")
        os.makedirs(args.output_dir)

    # Tensorboard
    tensorboard_writer = SummaryWriter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_archs = PUNCCAP_MODEL_MAPPING[args.model_arch]
    config = model_archs['config_clss'].from_pretrained(args.model_name_or_path,
                                                        num_plabels=len(PUNC_LABEL2ID),
                                                        num_clabels=len(CAP_LABEL2ID),
                                                        finetuning_task=args.task)
    model = model_archs['model_clss'].from_pretrained(args.model_name_or_path, config=config, from_tf=False)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # Trainer Summary
    total_params, trainable_param = get_total_model_parameters(model)
    LOGGER.info(f"{'=' * 20}TRAINER SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Parameters", "Values"])
    summary_table.add_rows([['Task', args.task],
                            ['Model architecture', args.model_arch],
                            ['Encoder name', args.model_name_or_path],
                            ['Total params', total_params],
                            ['Trainable params', trainable_param],
                            ['Max sequence length', args.max_seq_length],
                            ['Train batch size', args.train_batch_size],
                            ['Eval batch size', args.eval_batch_size],
                            ['Learning rate', args.learning_rate],
                            ['Number of epochs', args.epochs],
                            ['Weight decay', args.weight_decay],
                            ['Adam epsilon', args.adam_epsilon],
                            ['Max grad norm', args.max_grad_norm],
                            ['Early stop', args.early_stop],
                            ['Save step', args.save_step],
                            ['Use Cuda', not args.no_cuda]])
    LOGGER.info(summary_table)
    # Run
    best_score = 0.0
    cumulative_early_steps = 0
    trained_step = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Hey!!! Early stopping. Check your saved model.")
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Train
        train_loss = 0
        num_trainset = 0
        for fname, train_dataset in get_datasets(args.data_dir,
                                                 tokenizer,
                                                 max_seq_length=args.max_seq_length,
                                                 device=device,
                                                 use_crf=use_crf):
            train_sampler = RandomSampler(train_dataset)
            train_iterator = DataLoader(train_dataset,
                                        sampler=train_sampler,
                                        batch_size=args.train_batch_size,
                                        num_workers=0)
            trained_step += len(train_iterator)
            trainset_loss = train_one_epoch(model, optimizer, train_iterator, max_grad_norm=args.max_grad_norm)
            train_loss += trainset_loss
            num_trainset += 1
            tensorboard_writer.add_scalar(f'TRAIN_STEP/{fname}', trainset_loss, epoch)
            # Save checkpoint every train set to backup model
            if trained_step % args.save_step == 0:
                saved_file = Path(args.output_dir + f"/backup_model.pt")
                LOGGER.info(f"\t***Opps!!! Over save step, saving to {saved_file}...***")
                save_model(args, saved_file, model)

        tensorboard_writer.add_scalar('TRAIN_RESULT/Loss', train_loss / num_trainset, epoch)
        # Eval
        eval_f1_score = 0.0
        eval_acc = 0.0
        eval_loss = 0
        num_evalset = 0
        for fname, valid_dataset in get_datasets(args.data_dir,
                                                 tokenizer,
                                                 dtype='test',
                                                 max_seq_length=args.max_seq_length,
                                                 device=device,
                                                 use_crf=use_crf):
            valid_iterator = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
            evalset_loss, evalset_acc, evalset_f1_score = validate(model, valid_iterator, scheduler)
            eval_loss += evalset_loss
            eval_acc += evalset_acc
            eval_f1_score += evalset_f1_score
            num_evalset += 1
            tensorboard_writer.add_scalar(f'EVAL_STEP_LOSS/{fname}', evalset_loss, epoch)
            tensorboard_writer.add_scalar(f'EVAL_STEP_ACC/{fname}', evalset_acc, epoch)
            tensorboard_writer.add_scalar(f'EVAL_STEP_F1/{fname}', evalset_f1_score, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/Loss', eval_loss / num_evalset, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/Accuracy', eval_acc / num_evalset, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/F1-score', eval_f1_score / num_evalset, epoch)
        LOGGER.info(f"\tEpoch F1 score = {eval_f1_score} ; Best score = {best_score}")
        if eval_f1_score > best_score:
            best_score = eval_f1_score
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***Oh yeah!!! New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
            cumulative_early_steps = 0
        else:
            cumulative_early_steps += 1


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process... go go go!!!")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process... go go go!!!")
        test()
    else:
        LOGGER.error(
            f'[ERROR] - `{sys.argv[1]}` Are you kidding me? I only know `train` or `test`. Please read the README!!!!')
