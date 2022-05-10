from argparse import ArgumentParser


def get_test_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test'],
                        help='What processs to be run')
    parser.add_argument("--data_dir", default='datasets/News', type=str,
                        help="The input data dir. Should contain the .txt files (or other data files) for the task.")
    parser.add_argument("--overwrite_data", action='store_true',
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_train_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test'],
                        help='What processs to be run')
    parser.add_argument("--data_dir", default='datasets/News', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--overwrite_data", action='store_true',
                        help="Whether not to overwirte splitted dataset")
    parser.add_argument("--model_name_or_path", default='bert-base-multilingual-cased', type=str,
                        help="Pre-trained model selected in the list: bert-base-uncased, bert-base-cased...")
    parser.add_argument("--model_arch", default='lstm_crf', type=str,
                        help="Punctuation prediction model architecture selected in the list: original, crf, lstm_crf.")
    parser.add_argument("--output_dir", default='outputs/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=190, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--early_stop", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()