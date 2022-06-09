from argparse import ArgumentParser


def get_download_argument():
    parser = ArgumentParser()

    parser.add_argument('type', choices=['train', 'test', 'download'],
                        help='What processs to be run')
    parser.add_argument("--data_dir", default='datasets/', type=str,
                        help="The input data dir. The dir to saved downloaded file.")
    return parser.parse_args()


def get_build_dataset_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['build', 'split'],
                        help='What processs to be run')
    parser.add_argument("--corpus_path", default='datasets/Raw/corpus-full.txt', type=str,
                        help="The input corpus file path.")
    parser.add_argument("--split_test", action='store_true', default=False,
                        help="Whether not to split dataset to train and test set")
    parser.add_argument("--test_ratio", default=0.2, type=float,
                        help="It should be between 0.0 and 1.0 and represent the proportion of the dataset to include "
                             "in the test split")
    parser.add_argument("--truncate", action='store_true', default=False,
                        help="Truncate large dataset file to smaller files.")
    parser.add_argument("--truncate_size", default=150000000, type=int,
                        help="Maximun number of truncated file.")
    parser.add_argument("--skip_ratio", default=0.0, type=float,
                        help="It should be between 0.0 and 1.0 and represent the proportion of skip examples.")
    parser.add_argument('--num_worker', type=int, default=0,
                        help="How many subprocesses to use for data building. 0 means that the data will be loaded in "
                             "the main process. Only use when small corpus or more RAM")
    return parser.parse_args()


def get_split_argument():
    parser = ArgumentParser()

    parser.add_argument('type', choices=['build', 'split'],
                        help='What processs to be run')
    parser.add_argument("--data_dir", default='datasets/', type=str,
                        help="The input data dir. The dir to saved splited file.")
    parser.add_argument("--max_len", default=190, type=int,
                        help="The maximum total input sequence length.")

    return parser.parse_args()


def get_test_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'download'],
                        help='What processs to be run')
    parser.add_argument('--dataset_type', choices=['news', 'novels', 'custom'], nargs='?', const='custom', default='custom',
                        help='What dataset to be test')
    parser.add_argument("--data_dir", default='datasets/News', type=str,
                        help="The input data dir. Should contain the .txt files (or other data files) for the task.")
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--cached_dataset", action='store_true', default=False,
                        help="Whether not to cached converted dataset")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_worker', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in "
                             "the main process.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_train_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'download'],
                        help='What process to be run')
    parser.add_argument("--task", default='punctcap', type=str,
                        help="Training task selected in the list: punct, punctcap.")
    parser.add_argument("--data_dir", default='datasets/News', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--cached_dataset", action='store_true', default=False,
                        help="Whether not to cached converted dataset")
    parser.add_argument("--load_weights", default=None, type=str,
                        help='Path of pretrained file.')
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,
                        help="Pre-trained model selected in the list: bert-base-uncased, bert-base-cased...")
    parser.add_argument("--model_arch", default='lstm_crf', type=str,
                        help="Punctuation prediction model architecture selected in the list: lstm_softmax, lstm_crf, "
                             "bert_crf, bert_softmax.")
    parser.add_argument("--output_dir", default='outputs/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=190, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=5e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--early_stop", default=10.0, type=float,
                        help="")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_worker', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded "
                             "in the main process.")
    parser.add_argument('--save_step', type=int, default=20000,
                        help="")
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        help="Number of epochs with no improvement after which learning rate will be reduced. ")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()
