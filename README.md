<h1 align="center">🛠️VFastPunct-Trainer</h1>

Code by 🧑‍💻**Trong-Dat Ngo**.

Fast punctuation and capitalization restoration using Transformer Models for 🇻🇳Vietnamese

# Run
## Dataset preprocess
```bash
python vfastpunct/processor.py build --corpus_path datasets/corpus-full.txt --split_test --truncate
```
> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`build`, `split`].
> + ***corpus_path*** (`Union[str, os.PathLike]`, `*optional`): Training task selected in the list: [`punct`, `punctcap`]. Default: `punctcap`
> + ***split_test*** (`bool`, `*required`): The input data dir. Should contain the .csv files (or other data files) for the task.
> + ***test_ratio*** (`bool`, `*optional`) : Whether not to overwirte splitted dataset. Default=False
> + ***truncate*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***truncate_size*** (`str`, `*required`): Pre-trained model selected in the list: bert-base-uncased, bert-base-cased... Default=bert-base-multilingual-cased 
> + ***skip_ratio*** (`str`, `*required`): Pre-trained model selected in the list: bert-base-uncased, bert-base-cased... Default=bert-base-multilingual-cased 
>

## Train process
```bash
python vfastpunct/trainer.py train --model_arch lstm_softmax --model_name_or_path bert-base-multilingual-cased --learning_rate 1e-3 --max_seq_length 190 --epochs 100 --train_batch_size 32 --eval_batch_size 16 --data_dir datasets/ --output_dir outputs/
```

or

```bash
bash ./train.sh
```
> Arguments:
> + ***type*** (`str`,`*required`): What is process type to be run. Must in [`train`, `test`, `download`].
> + ***task*** (`str`, `*optional`): Training task selected in the list: [`punct`, `punctcap`]. Default: `punctcap`
> + ***data_dir*** (`Union[str, os.PathLike]`, `*required`): The input data dir. Should contain the .csv files (or other data files) for the task.
> + ***overwrite_data*** (`bool`, `*optional`) : Whether not to overwirte splitted dataset. Default=False
> + ***load_weights*** (`Union[str, os.PathLike]`, `*optional`): Path of pretrained file.
> + ***model_name_or_path*** (`str`, `*required`): Pre-trained model selected in the list: bert-base-uncased, bert-base-cased... Default=bert-base-multilingual-cased 
> + ***model_arch*** (`str`, `*required`): Punctuation prediction model architecture selected in the list: [`lstm_softmax`, `lstm_crf`, `bert_crf`, `bert_softmax`].
> + ***output_dir*** (`Union[str, os.PathLike]`, `*required`): The output directory where the model predictions and checkpoints will be written.
> + ***max_seq_length*** (`int`, `*optional`): The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Default=190.
> + ***train_batch_size*** (`int`, `*optional`): Total batch size for training. Default=32.
> + ***eval_batch_size*** (`int`, `*optional`): Total batch size for eval. Default=32.
> + ***learning_rate*** (`float`, `*optional`): The initial learning rate for Adam. Default=5e-5.
> + ***epochs*** (`float`, `*optional`): Total number of training epochs to perform. Default=100.0.
> + ***weight_decay*** (`float`, `*optional`): Weight deay if we apply some. Default=0.01.
> + ***adam_epsilon*** (`float`, `*optional`): Epsilon for Adam optimizer. Default=5e-8.
> + ***max_grad_norm*** (`float`, `*optional`): Max gradient norm. Default=1.0.
> + ***early_stop*** (`float`, `*optional`): Number of early stop step. Default=10.0.
> + ***no_cuda*** (`bool`, `*optional`): Whether not to use CUDA when available. Default=False.
> + ***seed*** (`bool`, `*optional`): Random seed for initialization. Default=42.
> + ***num_worker*** (`int`, `*optional`): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default=2.
> + ***save_step*** (`int`, `*optional`): The number of steps in the model will be saved. Default=10000.
> + ***scheduler_patience*** (`int`, `*optional`): Number of epochs with no improvement after which learning rate will be reduced. Default=2.
> + ***gradient_accumulation_steps*** (`int`, `*optional`): Number of updates steps to accumulate before performing a backward/update pass. Default=1.


## Test process
```bash

```

## Start Tensorboard
```bash
tensorboard --logdir runs --host 0.0.0.0 --port=6006
```

