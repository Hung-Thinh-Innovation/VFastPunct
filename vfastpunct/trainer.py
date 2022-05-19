from typing import Union, Generator
from tqdm import tqdm
from pathlib import Path
from vfastpunct.arguments import get_train_argument, get_test_argument
from vfastpunct.constants import LOGGER, PUNC_LABEL2ID, CAP_LABEL2ID
from vfastpunct.datasets import build_dataset, build_punccap_dataset
from vfastpunct.models import PuncBertLstmCrf, PuncCapBertLstmCrf, PuncCapBertConfig
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoTokenizer
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
                 device: str = 'cpu') -> Generator:
    for f in glob.glob(str(Path(ddir+f'/{dtype}_*_splitted.txt'))):
        LOGGER.info(f"Load file {f}")
        yield build_punccap_dataset(f, tokenizer, max_seq_length=max_seq_length, device=device)

def validate(model, valid_iterator, is_test=False):
    start_time = time.time()
    model.eval()
    eval_loss, nb_eval_steps = 0.0, 0.0
    eval_ppreds, eval_plabels = [], []
    eval_cpreds, eval_clabels = [], []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(valid_iterator), total=len(valid_iterator), position=0, leave=True):
            loss, eval_plogits, eval_clogits = model(**batch)
            eval_loss += loss.item()
            nb_eval_steps += 1
            active_accuracy = batch['label_masks'].view(-1) != 0
            plabels = torch.masked_select(batch['plabels'].view(-1), active_accuracy)
            clabels = torch.masked_select(batch['clabels'].view(-1), active_accuracy)
            eval_plabels.extend(plabels.cpu().tolist())
            eval_clabels.extend(clabels.cpu().tolist())
            eval_ppreds.extend(list(itertools.chain(*eval_plogits)))
            eval_cpreds.extend(list(itertools.chain(*eval_clogits)))
    epoch_loss = eval_loss / nb_eval_steps
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
        macro_avg = creports['macro avg']['f1-score'] + preports['macro avg']['f1-score']
        acc_avg = creports['accuracy'] + preports['accuracy']
        LOGGER.info(f"\tValidation Loss: {eval_loss}; "
                    f"Accuracy: {acc_avg}; "
                    f"Macro-F1 score: {macro_avg};"
                    f"Spend time: {time.time() - start_time}")
        return epoch_loss, acc_avg, macro_avg


def train_one_epoch(model, optimizer, train_iterator, max_grad_norm: float = 1.0):
    start_time = time.time()
    tr_loss, nb_tr_steps = 0.0, 0.0
    model.train()
    for idx, batch in tqdm(enumerate(train_iterator), total=len(train_iterator), position=0, leave=True):
        loss, _, _ = model(**batch)
        tr_loss += loss.item()
        nb_tr_steps += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = tr_loss / nb_tr_steps
    LOGGER.info(f"\tTraining Loss: {epoch_loss}; "
                f"Spend time: {time.time() - start_time}")
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

    config = AutoConfig.from_pretrained(configs.model_name_or_path, num_labels=len(checkpoint_data['classes']),
                                        finetuning_task="vipunc")
    model = PuncBertLstmCrf.from_pretrained(configs.model_name_or_path, config=config, from_tf=False)
    model.load_state_dict(checkpoint_data['model'])
    model.to(device)
    validate(model, test_iterator, is_test=True)


def train():
    best_score = 0.0
    args = get_train_argument()
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # train_dataset = build_dataset(args.data_dir,
    #                               tokenizer=tokenizer,
    #                               data_type='train',
    #                               max_seq_length=args.max_seq_length,
    #                               overwrite_data=args.overwrite_data,
    #                               device=device)
    #
    # valid_dataset = build_dataset(args.data_dir,
    #                               tokenizer=tokenizer,
    #                               data_type='valid',
    #                               max_seq_length=args.max_seq_length,
    #                               overwrite_data=args.overwrite_data,
    #                               device=device)
    #
    # train_sampler = RandomSampler(train_dataset)
    # train_iterator = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    # valid_iterator = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(PUNC_LABEL2ID),
    #                                     finetuning_task="vipunc")
    # model = PuncBertLstmCrf.from_pretrained(args.model_name_or_path, config=config, from_tf=False)
    config = PuncCapBertConfig.from_pretrained(args.model_name_or_path,
                                               num_plabels=len(PUNC_LABEL2ID),
                                               num_clabels=len(CAP_LABEL2ID),
                                               finetuning_task="vipunc")
    model = PuncCapBertLstmCrf.from_pretrained(args.model_name_or_path, config=config, from_tf=False)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    cumulative_early_steps = 0
    trained_step = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Hey!!! Early stopping. Check your saved model.")
        LOGGER.info(f"Training epoch {epoch}")
        for train_dataset in get_datasets(args.data_dir, tokenizer, max_seq_length=args.max_seq_length, device=device):
            train_sampler = RandomSampler(train_dataset)
            train_iterator = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                        num_workers=0)
            trained_step += len(train_iterator)
            train_one_epoch(model, optimizer, train_iterator, max_grad_norm=args.max_grad_norm)
            if trained_step > args.save_step:
                saved_file = Path(args.output_dir + f"/backup_model.pt")
                LOGGER.info(f"\t***Opps!!! Over save step, saving to {saved_file}...***")
                saved_data = {
                    'model': model.state_dict(),
                    'pclasses': PUNC_LABEL2ID,
                    'cclasses': CAP_LABEL2ID,
                    'args': args
                }
                torch.save(saved_data, saved_file)
                trained_step = 0
        eval_f1_score = 0.0
        for valid_dataset in get_datasets(args.data_dir, tokenizer, dtype='test',
                                          max_seq_length=args.max_seq_length, device=device):
            valid_iterator = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
            _, _, eval_f1_score = validate(model, valid_iterator)
        LOGGER.info(f"\tEpoch F1 score = {eval_f1_score} ; Best score = {best_score}")
        if eval_f1_score > best_score:
            best_score = eval_f1_score
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***Oh yeah!!! New best model, saving to {saved_file}...***")
            saved_data = {
                'model': model.state_dict(),
                'classes': PUNC_LABEL2ID,
                'args': args
            }
            torch.save(saved_data, saved_file)
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
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` Are you kidding me? I only know `train` or `test`. Please read the README!!!!')
