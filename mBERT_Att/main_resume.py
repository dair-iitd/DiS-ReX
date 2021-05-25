# coding:utf-8
import torch
import numpy as np
import json
import sys
import os
import argparse
import logging
from bert_encoder import BERTEntityEncoder
from inter_bag_attention import IntraBagAttention
from bag_re import BagRE

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt_resume', default='',
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc','p@10','p@30'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=2,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--optim', default='sgd', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=120, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
        help='Max number of training epochs')

args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
    raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

# Define the sentence encoder
sentence_encoder = BERTEntityEncoder(
    max_length=args.max_length,
    pretrain_path=args.pretrain_path,
    mask_entity=args.mask_entity
)


# Define the model
model = IntraBagAttention(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt='adamw',
    bag_size=128 // args.batch_size,
    warmup_step = 30000 // args.batch_size)


# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print(result)
# Print the result
print('Test set results:')
print('AUC: {}'.format(result['auc']))
print('Micro F1: {}'.format(result['micro_f1']))
print('P@10: {}'.format(result['p@10']))
print('P@30: {}'.format(result['p@30']))
