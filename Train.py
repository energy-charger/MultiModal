from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pandas as pd
from torch.nn import MSELoss
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import BertModel
import adapter
from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE
import torch.nn.functional as F
from Instance_supcon import SupConLoss
from SCL import SCL
from Unimodal_con import UnimodalConLoss
from prototype_supcon import ProtoConLoss

# from transformers import logging  # 11111111111111
# logging.set_verbosity_warning()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=60)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--model", type=str, default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--ca_layer", type=int, default=3)
parser.add_argument("--sa_layer", type=int, default=2)
parser.add_argument("--hidden_size", type=int, default=256, help="SA/CA hidden size")
parser.add_argument("--d_l", type=int, default=50, help="common dimension")
parser.add_argument("--ff_size", type=int, default=1024, help="FeedForwardNet size, keep its size == hidden_size * 4")
parser.add_argument("--multi_head", type=int, default=8, help="SA/CA multi_head in layer")
parser.add_argument("--hidden_size_head", type=int, default=32, help="hidden_size_head = hidden_size/multi_head")
parser.add_argument("--dropout_r", type=float, default=0.1, help="dropout rate for insert module")
parser.add_argument("--adapter_list1", default="12", type=str, help="The layer where add an adapter")
parser.add_argument("--adapter_list2", default="12", type=str, help="The layer where add an adapter")
parser.add_argument("--adapter_initializer_range", default=0.0002, type=float, help="adapter_initializer_weights_range")

args = parser.parse_args(args=[])
args.adapter_list1 = args.adapter_list1.split(',')
args.adapter_list1 = [int(i) for i in args.adapter_list1]
args.adapter_list2 = args.adapter_list2.split(',')
args.adapter_list2 = [int(i) for i in args.adapter_list2]


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)
        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            # print(inv_idx)
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])
        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)
        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_bert_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]
    # print(tokens)

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    # print(visual_zero.shape)
    # print(visual.shape)
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(input_ids)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    # print(segment_ids)
    # print(input_mask)

    pad_length = args.max_seq_length - len(input_ids)
    # print(pad_length)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    # print(input_ids)
    # print(input_mask)
    # print(segment_ids)
    return input_ids, visual, acoustic, input_mask, segment_ids


# def get_tokenizer(model):
#     return BertTokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = BertTokenizer.from_pretrained(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    pretrain_model = BertModel.from_pretrained(args.model)
    adapter_model = adapter.AdapterModel(args, pretrain_model.config, num_labels=1)
    model = (pretrain_model, adapter_model)

    pretrain_model.to(DEVICE)
    adapter_model.to(DEVICE)

    # Prepare optimizer
    param_optimizer1 = list(pretrain_model.named_parameters())
    param_optimizer2 = list(adapter_model.named_parameters())
    param_optimizer = param_optimizer1 + param_optimizer2  # 会包含adapter_model每一层的参数
    # param_optimizer = param_optimizer2
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def cosine(x, y):
    prod = torch.mul(x, y)
    normx = torch.norm(x)
    normy = torch.norm(y)
    cos = prod.div(torch.mul(torch.mul(normx, normy), 2))
    return 0.5 - cos


def get_causality_loss(x, x_useful, x_useless, labels):
    ranking_loss = torch.nn.SoftMarginLoss()
    batch_size = x.size(0)
    index = random.randint(0, batch_size - 1)
    anchor_label = (labels[index] > 0)  # positive:1 negative:0
    mask = (labels > 0)
    positive_mask = (mask == anchor_label).float()  # 与anchor标签相同的是正类
    negative_mask = (1 - positive_mask)  # 与anchor标签相反的是负类

    non_zeros1 = np.array(
        [i for i, e in enumerate(positive_mask) if e != 0 and i != index])

    non_zeros2 = np.array(
        [i for i, e in enumerate(negative_mask) if e != 0 and i != index])

    anchor = x[index]
    anchor_useful = x_useful[index]
    anchor_useless = x_useless[index]

    seed1 = random.randint(0, non_zeros1.size - 1)
    positive = x[non_zeros1[seed1]] * positive_mask[non_zeros1[seed1]]
    positive_useful = x_useful[non_zeros1[seed1]] * positive_mask[non_zeros1[seed1]]
    positive_useless = x_useless[non_zeros1[seed1]] * positive_mask[non_zeros1[seed1]]

    seed2 = random.randint(0, non_zeros2.size - 1)
    negative = x[non_zeros2[seed2]] * negative_mask[non_zeros2[seed2]]
    negative_useful = x_useful[non_zeros2[seed2]] * negative_mask[non_zeros2[seed2]]
    negative_useless = x_useless[non_zeros2[seed2]] * negative_mask[non_zeros2[seed2]]

    target = torch.ones_like(anchor)

    restitution1 = ranking_loss(cosine(anchor_useful, positive_useful) - cosine(anchor, positive), target) \
                   + ranking_loss(cosine(anchor, negative) - cosine(anchor_useful, negative_useful), target)
    restitution2 = ranking_loss(cosine(anchor, positive) - cosine(anchor_useless, positive_useless), target) \
                   + ranking_loss(cosine(anchor_useless, negative_useless) - cosine(anchor, negative), target)

    return restitution1 + restitution2


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    pretrain_model = model[0]
    adapter_model = model[1]
    pretrain_model.train()
    adapter_model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        # visual = torch.squeeze(visual, 1)
        # acoustic = torch.squeeze(acoustic, 1)
        pretrain_model_outputs = pretrain_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        # print(type(pretrain_model_outputs))
        # print(pretrain_model_outputs[0].shape)
        # print(pretrain_model_outputs[1].shape)
        # print(len(pretrain_model_outputs[3]))
        # print(len(pretrain_model_outputs[2]))
        # outputs, adapter_states_1, adapter_states_1_useful, adapter_states_1_useless, \
        # adapter_states_2, adapter_states_2_useful, adapter_states_2_useless = adapter_model(pretrain_model_outputs,
        #                                                                                     visual, acoustic,
        #                                                                                     labels=None)
        outputs, adapter_states_1, adapter_states_2, fusion_feature, output_l = adapter_model(pretrain_model_outputs,
                                                                                              visual,
                                                                                              acoustic,
                                                                                              labels=None)
        # outputs, hidden_states_last = adapter_model(pretrain_model_outputs, visual, acoustic, labels=None)

        logits = outputs[0]

        label_7 = label_ids.view(label_ids.shape[0], 1)
        label_7 = torch.round(label_7)
        device = torch.device('cuda')

        one = torch.tensor(1.0).to(device)
        none = torch.tensor(-1.0).to(device)
        label_2 = label_ids.view(label_ids.shape[0], 1)
        label_2 = torch.where(label_2 > 0, one, label_2)
        label_2 = torch.where(label_2 < 0, none, label_2)

        label_used = label_7

        loss_fct = MSELoss()
        loss_con = SupConLoss(temperature=0.7)
        # loss_SCL = SCL()
        loss_uni = UnimodalConLoss()
        loss_proto = ProtoConLoss()
        loss_supcon = loss_con(adapter_states_1, adapter_states_2, fusion_feature, label_used)
        # loss_SCLf = loss_SCL(adapter_states_1, adapter_states_2)
        loss_unif = loss_uni(adapter_states_1, adapter_states_2, output_l, label_used)
        loss_protof = loss_proto(fusion_feature, label_used)

        # loss_causality = 0.002 * get_causality_loss(adapter_states_1, adapter_states_1_useful,
        #                                             adapter_states_1_useless, label_ids.view(-1)) \
        #                  + 0.002 * get_causality_loss(adapter_states_2, adapter_states_2_useful,
        #                                               adapter_states_2_useless, label_ids.view(-1)) \
        #                  + 0.001 * loss_fct(adapter_states_1_useful, label_ids.view(-1)) \
        #                  + 0.001 * loss_fct(adapter_states_2_useful, label_ids.view(-1))

        # loss = loss_fct(logits.view(-1), label_ids.view(-1)) + loss_causality
        loss = loss_fct(logits.view(-1), label_ids.view(-1)) + loss_protof + loss_unif

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            pretrain_model.zero_grad()
            adapter_model.zero_grad()
            # optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    pretrain_model = model[0]
    adapter_model = model[1]
    # model.eval()
    pretrain_model.eval()
    adapter_model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            pretrain_model_outputs = pretrain_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            # outputs, adapter_states_1, adapter_states_1_useful, adapter_states_1_useless, \
            # adapter_states_2, adapter_states_2_useful, adapter_states_2_useless = adapter_model(pretrain_model_outputs,
            #                                                                                     visual, acoustic,
            #                                                                                     labels=None)
            #
            # logits = outputs[0]
            #
            # loss_fct = MSELoss()
            #
            # loss_causality = 0.002 * get_causality_loss(adapter_states_1, adapter_states_1_useful,
            #                                             adapter_states_1_useless, label_ids.view(-1)) \
            #                  + 0.002 * get_causality_loss(adapter_states_2, adapter_states_2_useful,
            #                                               adapter_states_2_useless, label_ids.view(-1)) \
            #                  + 0.001 * loss_fct(adapter_states_1_useful, label_ids.view(-1)) \
            #                  + 0.001 * loss_fct(adapter_states_2_useful, label_ids.view(-1))
            #
            # loss = loss_fct(logits.view(-1), label_ids.view(-1)) + loss_causality
            outputs, adapter_states_1, adapter_states_2, fusion_feature = adapter_model(pretrain_model_outputs, visual,
                                                                                        acoustic,
                                                                                        labels=None)
            logits = outputs[0]

            label_7 = label_ids.view(label_ids.shape[0], 1)
            label_7 = torch.round(label_7)

            label_2 = label_ids.view(label_ids.shape[0], 1)
            label_2 = torch.where(label_2 > 0, 1, label_2)
            label_2 = torch.where(label_2 < 0, -1, label_2)

            label_used = label_2
            loss_fct = MSELoss()
            loss_con = SupConLoss(temperature=0.7)
            loss_supcon = loss_con(adapter_states_1, adapter_states_2, fusion_feature, label_used)

            loss = loss_fct(logits.view(-1), label_ids.view(-1)) + loss_supcon

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def tst_epoch(model: nn.Module, test_dataloader: DataLoader):
    # model.eval()
    pretrain_model = model[0]
    adapter_model = model[1]
    pretrain_model.eval()
    adapter_model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            pretrain_model_outputs = pretrain_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            # outputs, adapter_states_1, adapter_states_1_useful, adapter_states_1_useless, \
            # adapter_states_2, adapter_states_2_useful, adapter_states_2_useless = adapter_model(pretrain_model_outputs,
            #                                                                                     visual, acoustic,
            #                                                                                     labels=None)
            outputs, adapter_states_1, adapter_states_2, fusion_feature, output_l = adapter_model(
                pretrain_model_outputs, visual,
                acoustic,
                labels=None)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def tst_score_model(model: nn.Module, test_dataloader: DataLoader):
    preds, y_test = tst_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0])

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(preds_a7, truth_a7)

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    # print(np.corrcoef(preds, y_test))

    # negative/positive
    binary_preds = (preds[non_zeros] > 0)
    binary_truth = (y_test[non_zeros] > 0)

    f_score_np = f1_score(binary_truth, binary_preds, average="weighted")
    binary_acc_np = accuracy_score(binary_truth, binary_preds)

    # negative/non_negative
    binary_preds = preds >= 0
    binary_truth = y_test >= 0

    f_score_nn = f1_score(binary_truth, binary_preds, average="weighted")
    binary_acc_nn = accuracy_score(binary_truth, binary_preds)

    return binary_acc_np, binary_acc_nn, mult_a7, mae, corr, f_score_np, f_score_nn


def train(model, train_dataloader, validation_dataloader, test_data_loader, optimizer, scheduler, ):
    valid_losses = []
    test_acc2s_np = []
    test_acc2s_nn = []
    test_acc7s = []
    test_maes = []
    test_corrs = []
    test_f_score_nps = []
    test_f_score_nns = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        # valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc2_np, test_acc2_nn, test_acc7, test_mae, test_corr, test_f_score_np, test_f_score_nn = tst_score_model(
            model, test_data_loader)

        print(
            "epoch:{}, train_loss:{},  test_acc2_np:{}, test_acc2_nn:{}, test_acc7:{}".format(
                epoch_i, train_loss, test_acc2_np, test_acc2_nn, test_acc7
            )
        )

        # valid_losses.append(valid_loss)
        test_acc2s_np.append(test_acc2_np)
        test_acc2s_nn.append(test_acc2_nn)
        test_acc7s.append(test_acc7)
        test_maes.append(test_mae)
        test_corrs.append(test_corr)
        test_f_score_nps.append(test_f_score_np)
        test_f_score_nns.append(test_f_score_nn)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    # "valid_loss": valid_loss,
                    "test_acc2_np": test_acc2_np,
                    "test_acc2_nn": test_acc2_nn,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score_np": test_f_score_np,
                    "test_f_score_nn": test_f_score_nn,
                    "best_test_acc2_np": max(test_acc2s_np),
                    "best_test_acc2_nn": max(test_acc2s_nn),
                    "best_test_acc7": max(test_acc7s),
                    "best_test_mae": min(test_maes),
                    "best_test_corr": max(test_corrs)
                }
            )
        )

    print(
        "best_test_acc2_np:{}, best_test_acc2_nn:{}, best_test_acc7:{}".format(
            max(test_acc2s_np), max(test_acc2s_nn), max(test_acc7s)
        )
    )


def main():
    wandb.init(project="bert_GRU_SNR")
    wandb.config.update(args)

    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
