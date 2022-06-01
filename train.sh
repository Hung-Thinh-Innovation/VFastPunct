#!/usr/bin/env bash

pip install -r requirements.txt
export PYTHONPATH=.
python vfastpunct/trainer.py train --model_arch lstm_softmax --model_name_or_path bert-base-multilingual-cased --learning_rate 1e-3 --max_seq_length 190 --epochs 100 --train_batch_size 32 --eval_batch_size 16 --data_dir datasets/ --output_dir outputs/