
# -*- coding: utf-8 -*-
import re
import torch
import pandas as pd

from torch import nn
from os.path import join as pjoin
from plm import LightningPLM
from utils.model_util import load_model

def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 4)
    args.log = getattr(args, 'log', True)

def tokenize(tokenizer, text, max_len):
    q_toked = tokenizer.tokenize(tokenizer.cls_token + text + tokenizer.sep_token)
    if len(q_toked) > max_len:
        q_toked = q_toked[:max_len-1] + [q_toked[-1]]

    token_ids = tokenizer.convert_tokens_to_ids(q_toked)
    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_len:
        token_ids += [tokenizer.pad_token_id]
        attention_mask += [0]

    return token_ids, attention_mask


def eval_user_input(args, model, tokenizer, device):

    def is_valid(query: str) -> bool:
        return re.sub('[\s]*', '', query)

    query = input('사용자 입력: ')
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        while is_valid(query):

            input_ids, attention_mask = tokenize(tokenizer, text=query, max_len=args.max_len)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(device=device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits.detach().cpu()
            probs = softmax(logits)
            # print(probs, torch.argmax(probs, dim=-1))

            pred = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
            prob = torch.max(probs).numpy().tolist()

            print(f"Query: {query}")
            print("Predict: {} ({:.2f})".format(pred[0], prob))

            query = input('사용자 입력: ')
            

def eval_test_set(args, model, tokenizer, device):
    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    test_data = test_data.dropna(axis=0)

    pred_list = []
    count = 0

    with torch.no_grad():
        for row in test_data.iterrows():

            utterance = row[1]['proc_text']
            label = int(row[1]['label'])
            
            input_ids, attention_mask = tokenize(tokenizer, text=utterance, max_len=args.max_len)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(device=device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits.detach().cpu()

            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            print(predictions[0], utterance)
            pred_list.append(predictions[0]) 

            if predictions[0] == label:
                count += 1

        test_data['pred'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'{args.model_name}-{round(count/len(test_data), 2)*100}.csv'), index=False)
        print(f"Accuracy: {count/len(test_data)}")
            

def evaluation(args, **kwargs):
    # load params
    base_setting(args)
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    print(args.model_pt)

    model, tokenizer = load_model(args.model_type, args.num_labels)
    model = model.cuda()

    if args.model_pt is not None:
        if args.model_pt.endswith('ckpt'):
            model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
        else:
            raise TypeError('Unknown file extension')

    model = model.cuda()     
    model.eval()

    if args.user_input:
        eval_user_input(args, model, tokenizer, device)
    else:
        eval_test_set(args, model, tokenizer, device)