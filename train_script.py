"""
to be done, properly, it's 3 a.m.,
I am more than confident, that any button pushed in this time is foolish
"""
from models import Encoder, Classifier, TokenEncoder, CLEFModel
from datasets import AudioConfig, OnlineDataset
from trainer import CustomTrainer

import warnings
warnings.filterwarnings('ignore')

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import json
with open("train_cfg.json", 'r', encoding="utf-8" ) as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    #==dataset-kwargs
    csv_path = config["csv_path"]
    os.environ['WANDB_API_KEY'] = config["WANDB_API_KEY"]
    train_size = config["train_size"]
    eval_size = config["eval_size"]
    wav_path = config["wav_path"]

    #==audio-kwargs
    n_fft = config["n_fft"]
    win_length = config["win_length"]
    hop_length = config["hop_length"]
    #slicing_func = encoder.stripe_w_overlap,
    n_mels = config["n_mels"]
    stripe_width = config["stripe_width"]
    stripe_overlap = config["stripe_width"]
    prior_cut_sec = config["prior_cut_sec"]

    #==model-kwargs
    in_classifier_num_head = config["in_classifier_num_head"]
    d_c = config["d_c"]
    d_c1 = config["d_c1"]
    d_r = config["d_r"]
    pool_type = config["pool_type"]

    #==train-kwargs
    batch_size = config["batch_size"]
    lr_max = config["lr_max"]
    lr_min = config["lr_min"]
    wandb_name = config["csv_path"]
    epochs = config["epochs"]
    eval_freq = config["eval_freq"]
    optim_freq = config["optim_freq"]
    save_freq = config["save_freq"]
    skip_seq_len = config["skip_seq_len"]
    micro_batch_size = config["micro_batch_size"]

    #==csv
    train_df = pd.read_csv(csv_path)

    #todo statify k-fold
    le = LabelEncoder().fit(train_df.primary_label)
    train_idx, small_test_idx = train_test_split(
        np.arange(len(train_df)),
        train_size = train_size,
        test_size = eval_size,
        random_state = 32,
        stratify = train_df['primary_label']
    )

    #==model-definition
    encoder = Encoder('efficientnet_b3a')
    config = AudioConfig(
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        slicing_func=encoder.stripe_w_overlap,
        n_mels = n_mels,
        stripe_width = stripe_width,
        stripe_overlap = stripe_overlap,
        prior_cut_sec = prior_cut_sec,
    )
    encoder.set_audiopreprocessing(config)
    train_ds = OnlineDataset(
        meta_df = train_df.iloc[train_idx],
        label_encoder = le,
        config = config,
        group_mode = True,
        return_id = False,
        root_dir = wav_path,
    )
    eval_ds = OnlineDataset(
        meta_df = train_df.iloc[small_test_idx],
        label_encoder = le,
        config = config,
        group_mode = True,
        return_id = False,
        multitarget = False, #todo, how to properly handle multitarget in eval?!
        root_dir=wav_path,
    )

    single_target_head = TokenEncoder(
        embedding_size = encoder.backbone.num_features,
        class_num = len(le.classes_),
        num_head = in_classifier_num_head,
        d_c = d_c,
        d_c1 = d_c1,
        d_rotate = d_r,
        pool_type = pool_type,
    )
    # multi_target_head = TokenEncoder(
    #     embedding_size = encoder.backbone.num_features,
    #     class_num = len(le.classes_),
    # )
    classifier = Classifier(
        seq_mode = True,
        single_head = single_target_head,
        #multi_head = multi_target_head, #todo
        multi_head = single_target_head, #todo same model is trained in different modes
        single_activation = nn.Softmax(dim = -1), #dim = -1, because tokenwise softmax
        multi_activation= nn.Sigmoid(),
    )
    model = CLEFModel(
        encoder = encoder,
        classifier = classifier,
        padding_value = 0,
    ).to(device)

    #==train-def
    optimizer = optim.AdamW(
        params = model.parameters(),
        lr = lr_max
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer = optimizer,
        T_0 = int(len(train_ds)//batch_size*0.5), #change with respect to trainer
        eta_min = lr_min,
    )
    loss = nn.BCELoss(reduction = 'sum')
    trainer = CustomTrainer(
        optimizer = optimizer,
        scheduler = scheduler,
        loss = loss,
        batch_size = batch_size,
    )

    #==train-routine
    wandb.init(project="my-project", name=wandb_name)

    model = trainer.train(
        model = model,
        train_ds = train_ds,
        eval_ds = eval_ds,
        epochs = epochs,
        #steps = 10_000,
        eval_freq = eval_freq,
        optim_freq = optim_freq,
        save_freq = save_freq,
        skip_seq_len = skip_seq_len,
        micro_batch_size = micro_batch_size,
    )
    wandb.finish()