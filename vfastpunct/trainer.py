from vfastpunct.arguments import get_train_argument, get_test_argument, get_download_argument
from vfastpunct.constants import LOGGER, MODEL_MAPPING, PUNCT_LABEL2ID, CAP_LABEL2ID
from vfastpunct.datasets import build_and_cached_dataset
from vfastpunct.ultis import get_total_model_parameters, download_dataset_from_drive

from typing import Union, Generator
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler

import os
import gc
import glob
import sys
import time
import torch
import random
import numpy as np
import itertools


def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_datasets(args, tokenizer, dtype: str = 'train',  use_crf: bool = False, device: str = 'cpu') -> Generator:
    data_files = glob.glob(str(Path(args.data_dir + f'/{dtype}_*_splitted.txt')))
    LOGGER.info(f"Found {len(data_files)} to {dtype.upper()} in {args.data_dir}")
    for fpath in data_files:
        fname = os.path.basename(fpath)
        dataset = build_and_cached_dataset(fpath,
                                           tokenizer,
                                           args.task,
                                           max_seq_length=args.max_seq_length,
                                           use_crf=use_crf,
                                           cached_dataset=args.cached_dataset,
                                           overwrite_data=args.overwrite_data,
                                           device=device)
        yield len(data_files), fname, dataset


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'pclasses': PUNCT_LABEL2ID,
        'cclasses': CAP_LABEL2ID,
        'args': args
    }
    torch.save(saved_data, saved_file)


def validate(args,
             model,
             tokenizer,
             cur_epoch: int = 0,
             device: str = 'cpu',
             use_crf: bool = False,
             is_test=False,
             tb_writer=None):
    start_time = time.time()
    model.eval()
    eval_loss, eval_ploss, eval_closs, nb_eval_steps = 0.0, 0.0, 0.0, 0
    eval_ppreds, eval_plabels, eval_cpreds, eval_clabels = [], [], [], []
    evaled_sets = 0
    for num_of_sets, fname, valid_dataset in get_datasets(args, tokenizer, 'test', use_crf=use_crf, device=device):
        step_loss, step_ploss, step_closs = 0.0, 0.0, 0.0
        evaled_sets += 1
        valid_iterator = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
        # Run one step on sub-dataset
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_iterator), total=len(valid_iterator), desc=f'[EVAL]Epoch {cur_epoch}/{args.epochs}; Sub-dataset{evaled_sets}/{num_of_sets}', position=0, leave=True):
                outputs = model(**batch)
                step_loss += outputs.loss.detach().item()
                step_ploss += outputs.ploss.detach().item()
                step_closs += outputs.closs.detach().item()
                nb_eval_steps += 1
                active_accuracy = batch['label_masks'].view(-1) != 0
                plabels = torch.masked_select(batch['plabels'].view(-1), active_accuracy)
                eval_plabels.extend(plabels.detach().cpu().tolist())
                if 'clabels' in batch:
                    clabels = torch.masked_select(batch['clabels'].view(-1), active_accuracy)
                    eval_clabels.extend(clabels.detach().cpu().tolist())
                if isinstance(outputs.ptags[-1], list):
                    eval_ppreds.extend(list(itertools.chain(*outputs.ptags)))
                    eval_cpreds.extend(list(itertools.chain(*outputs.ctags)))
                else:
                    eval_ppreds.extend(outputs.ptags)
                    eval_cpreds.extend(outputs.ctags)
        eval_loss += step_loss
        eval_ploss += step_ploss
        eval_closs += step_closs
        if tb_writer is not None:
            preports = classification_report(eval_plabels, eval_ppreds, output_dict=True, zero_division=0)
            step_avg_f1 = preports['macro avg']['f1-score']
            step_avg_acc = preports['accuracy']
            if len(eval_cpreds) > 0:
                creports = classification_report(eval_clabels, eval_cpreds, output_dict=True, zero_division=0)
                step_avg_f1 = (step_avg_f1 + creports['macro avg']['f1-score']) / 2
                step_avg_acc = (step_avg_acc + creports['accuracy']) / 2
            tb_writer.add_scalar(f'EVAL_STEP_LOSS/{fname}', (step_loss / len(valid_iterator)), cur_epoch)
            tb_writer.add_scalar(f'EVAL_STEP_ACC/{fname}', step_avg_acc,cur_epoch)
            tb_writer.add_scalar(f'EVAL_STEP_F1/{fname}', step_avg_f1, cur_epoch)
    epoch_loss = eval_loss / nb_eval_steps
    epoch_ploss = eval_ploss / nb_eval_steps
    epoch_closs = eval_closs / nb_eval_steps
    if is_test:
        preports = classification_report(eval_plabels, eval_ppreds, zero_division=0)
        LOGGER.info(f'\tTest Loss: {eval_loss}; Spend time: {time.time() - start_time}')
        LOGGER.info('Punct Report:')
        LOGGER.info(preports)
        if len(eval_cpreds) > 0:
            creports = classification_report(eval_clabels, eval_cpreds, zero_division=0)
            LOGGER.info('Cap Report:')
            LOGGER.info(creports)
        return epoch_loss
    else:
        preports = classification_report(eval_plabels, eval_ppreds, output_dict=True, zero_division=0)
        epoch_avg_f1 = preports['macro avg']['f1-score']
        epoch_avg_acc = preports['accuracy']
        creports = {'accuracy': 0.0, 'macro avg': {'f1-score': 0.0}}
        if len(eval_cpreds) > 0:
            creports = classification_report(eval_clabels, eval_cpreds, output_dict=True, zero_division=0)
            epoch_avg_f1 = (epoch_avg_f1 + creports['macro avg']['f1-score']) / 2
            epoch_avg_acc = (epoch_avg_acc + creports['accuracy']) / 2
        LOGGER.info(f"\t{'*' * 20}Validate Summary{'*' * 20}")
        LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f} (pLoss: {epoch_ploss:.4f}; cLoss: {epoch_closs:.4f});\n"
                    f"\tAccuracy: {epoch_avg_acc:.4f} (pAccuracy: {preports['accuracy']:.4f}; cAccuracy: {creports['accuracy']:.4f});\n"
                    f"\tMacro-F1 score: {epoch_avg_f1:.4f} (pF1: {preports['macro avg']['f1-score']:.4f}; cF1: {creports['macro avg']['f1-score']:.4f});\n"
                    f"\tSpend time: {time.time() - start_time}")
        eval_ppreds, eval_plabel, eval_cpreds, eval_clabels = None, None, None, None
        return epoch_loss, epoch_avg_acc, epoch_avg_f1


def train_one_epoch(args,
                    cur_epoch: int,
                    model,
                    optim,
                    tokenizer,
                    max_grad_norm: float = 1.0,
                    device: str = 'cpu',
                    use_crf: bool= False,
                    tb_writer=None):
    start_time = time.time()
    tr_loss, nb_tr_steps = 0.0, 0.0
    model.train()
    trained_set = 0
    for num_of_sets, fname, train_dataset in get_datasets(args, tokenizer, 'train', use_crf=use_crf, device=device):
        train_sampler = RandomSampler(train_dataset)
        train_iterator = DataLoader(train_dataset,
                                    sampler=train_sampler,
                                    batch_size=args.train_batch_size,
                                    num_workers=0)
        step_loss = 0.0
        trained_set += 1
        tqdm_desc = f'[TRAIN]Epoch {cur_epoch}/{args.epochs}; Sub-dataset {trained_set}/{num_of_sets}'
        tqdm_bar = tqdm(enumerate(train_iterator), total=len(train_iterator), desc=tqdm_desc, position=0, leave=True)
        for idx, batch in tqdm_bar:
            outputs = model(**batch)
            # backward pass
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optim.zero_grad()
            outputs.loss.backward()
            optim.step()
            tr_loss += outputs.loss.detach().item()
            step_loss += outputs.loss.detach().item()
            nb_tr_steps += 1
            # Save checkpoint to backup model
            if nb_tr_steps % args.save_step == 0:
                saved_file = Path(args.output_dir + f"/backup_model.pt")
                LOGGER.info(f"\t***Opps!!! Over save step, saving to {saved_file}...***")
                save_model(args, saved_file, model)
        if tb_writer is not None:
            tb_writer.add_scalar(f'TRAIN_STEP/{fname}', step_loss, cur_epoch)
    epoch_loss = tr_loss / nb_tr_steps
    LOGGER.info(f"\t{'*' * 20}Train Summary{'*' * 20}")
    LOGGER.info(
        f"\tTraining Lr: {optim.param_groups[0]['lr']}; Loss: {epoch_loss:.4f}; Spend time: {time.time() - start_time}")
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


def train():
    args = get_train_argument()
    set_ramdom_seed(args.seed)
    use_crf = True if 'crf' in args.model_arch else False
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found! Where is dataset ??????'

    if not os.path.isdir(args.output_dir):
        LOGGER.info(f"Create saved directory {args.output_dir}")
        os.makedirs(args.output_dir)

    # Tensorboard
    tensorboard_writer = SummaryWriter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_archs = MODEL_MAPPING[args.task][args.model_arch]
    config = model_archs['config_clss'].from_pretrained(args.model_name_or_path,
                                                        num_plabels=len(PUNCT_LABEL2ID),
                                                        num_clabels=len(CAP_LABEL2ID),
                                                        finetuning_task=args.task)
    model = model_archs['model_clss'].from_pretrained(args.model_name_or_path, config=config, from_tf=False)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.load_weights is not None:
        LOGGER.info(f'Load pretrained model weights from "{args.load_weights}"')
        if device == 'cpu':
            checkpoint_data = torch.load(args.load_weights, map_location='cpu')
        else:
            checkpoint_data = torch.load(args.load_weights)
        model.load_state_dict(checkpoint_data['model'])
        del checkpoint_data
        gc.collect()

    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.scheduler_patience)

    # Trainer Summary
    total_params, trainable_param = get_total_model_parameters(model)
    LOGGER.info(f"{'=' * 20}TRAINER SUMMARY{'=' * 20}")
    summary_table = PrettyTable(["Parameters", "Values"])
    summary_table.add_rows([['Task', args.task],
                            ['Model architecture', args.model_arch],
                            ['Encoder name', args.model_name_or_path],
                            ['Load weights', args.load_weights],
                            ['Total params', total_params],
                            ['Trainable params', trainable_param],
                            ['Max sequence length', args.max_seq_length],
                            ['Train batch size', args.train_batch_size],
                            ['Eval batch size', args.eval_batch_size],
                            ['Number of workers', args.num_worker],
                            ['Learning rate', args.learning_rate],
                            ['Number of epochs', args.epochs],
                            ['Weight decay', args.weight_decay],
                            ['Scheduler patience', args.scheduler_patience],
                            ['Adam epsilon', args.adam_epsilon],
                            ['Max grad norm', args.max_grad_norm],
                            ['Early stop', args.early_stop],
                            ['Save step', args.save_step],
                            ['Use Cuda', not args.no_cuda],
                            ['Cached dataset', args.cached_dataset]])
    LOGGER.info(summary_table)
    # Run
    best_score = 0.0
    cumulative_early_steps = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Hey!!! Early stopping. Check your saved model.")
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Fit model with dataset
        train_loss = train_one_epoch(args,
                                     cur_epoch=epoch,
                                     model=model,
                                     optim=optimizer,
                                     tokenizer=tokenizer,
                                     max_grad_norm=args.max_grad_norm,
                                     device=device,
                                     use_crf=use_crf,
                                     tb_writer=tensorboard_writer)
        tensorboard_writer.add_scalar('TRAIN_RESULT/Loss', train_loss, epoch)
        gc.collect()
        # Eval
        eval_loss, eval_acc, eval_f1 = validate(args,
                                                cur_epoch=epoch,
                                                model=model,
                                                tokenizer=tokenizer,
                                                device=device,
                                                use_crf=use_crf,
                                                tb_writer=tensorboard_writer)
        if scheduler is not None:
            scheduler.step(eval_f1)
        tensorboard_writer.add_scalar('EVAL_RESULT/Loss', eval_loss, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/Accuracy', eval_acc, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/F1-score', eval_f1, epoch)
        LOGGER.info(f"\tEpoch F1 score = {eval_f1} ; Best score = {best_score}")
        if eval_f1 > best_score:
            best_score = eval_f1
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***Oh yeah!!! New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
            cumulative_early_steps = 0
        else:
            cumulative_early_steps += 1
        gc.collect()


def download():
    args = get_download_argument()
    download_dataset_from_drive(args.data_dir)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process... go go go!!!")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process... go go go!!!")
        test()
    elif sys.argv[1] == 'download':
        LOGGER.info("Start Download process... go go go!!!")
        download()
    else:
        LOGGER.error(
            f'[ERROR] - `{sys.argv[1]}` Are you kidding me? I only know `train`, `test` or `download`. Please read '
            f'the README!!!!')
