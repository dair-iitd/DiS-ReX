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
parser.add_argument('--ckpt', default='mnre_baseline',
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc','p@10','p@30'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='../disrex_dataset/disrex_train.txt', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='../disrex_dataset/disrex_val.txt', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='../disrex_dataset/disrex_test.txt', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='../disrex_dataset/rel2id.txt', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=2,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--optim', default='adamw', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=120, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=5, type=int,
        help='Max number of training epochs')
parser.add_argument('--num_lang', default=4, type=int,
        help='Max number of training epochs')


parser.add_argument('--finetune', default='False', type=str,
        help='finetune of relx?')
parser.add_argument('--rel2id_new_file', default=None, type=str,
        help='path to new rel2id file for finetune')
parser.add_argument('--save_name', default='', type=str,
        help='name for saving checkpoint')
parser.add_argument('--seed', default=1, type=int,
        help='random seed')
args = parser.parse_args()
import os
import random
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
    raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
if args.finetune=='True':
        rel2id_new = json.load(open(args.rel2id_new_file))

# Define the sentence encoder
sentence_encoder = BERTEntityEncoder(
    max_length=args.max_length,
    pretrain_path=args.pretrain_path,
    mask_entity=args.mask_entity
)


# Define the model
model = IntraBagAttention(sentence_encoder, len(rel2id), rel2id)
if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt)['state_dict'])
if args.finetune=='True':
        model = IntraBagAttention(model.sentence_encoder,len(rel2id_new),rel2id_new)
        print(model)

# Define the whole training framework
framework = BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    num_lang = args.num_lang,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt='adamw',
    bag_size=args.bag_size,
    warmup_step = 30000 // args.batch_size)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test
# framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print(result)
# Print the result
print('Test set results:')
print('AUC: {}'.format(result['auc']))
print('Micro F1: {}'.format(result['micro_f1']))
print('P@10: {}'.format(result['p@10']))
print('P@30: {}'.format(result['p@30']))
