#!/usr/bin/env bash

pip install -r requirements.txt
export PYTHONPATH=.
python vfastpunct/trainer.py test --data_dir datasets/ --batch_size 512  --model_path outputs/best_model.pt --num_worker 20 --cached_dataset