# importing csv module
import csv
import os
import ast
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from torch.functional import F
from model import GPTConfig, GPT
import numpy as np


init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype='float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# folder path
dir_path = r'evaluation_data'

def load_model(out_dir):
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    return model

def get_ppl(model, start):
    # ok let's assume gpt-2 encodings by default
    # print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    start_ids = encode(start)
    x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...])
    y = (torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            logits, loss = model(x, y)
            return loss.exp().item()


def score_file(filename, models):
    score_right = 0
    score_all = 0
    # initializing the titles and rows list
    rows = []

    # reading csv file
    with open(dir_path+'/'+filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        # get total number of rows
        #print("Total no. of rows: %d" % (csvreader.line_num))


    for row in rows[3:]:
        stability = row[5]
        print("stability: "+str(stability))
        peak_months=[]
        if stability=='0':
            peak_months=ast.literal_eval(row[6])

        print("peak_months: ")
        print(peak_months)

        # parsing each column of a row
        for col in row[:5]:
            print(col)
            #run models and get a list of ppl for each month
            if peak_months==[]:
                ppls = []
                for model in models:
                    ppl = get_ppl(model, col)
                    print(ppl)
                    ppls.append(ppl)
                avg = np.average(ppls)
                std = np.std(ppls)
                rstd = std/avg
                #if rstd <0.25
                if rstd < 0.25:
                    score_right+=1
                    score_all+=1
                else:
                    score_all+=1
            else:
                ppls = []
                for model in models:
                    ppl = get_ppl(model, col)
                    print(ppl)
                    ppls.append(ppl)
                #get minimum index
                min_index = ppls.index(min(ppls))
                print("min_index: "+str(min_index))
                if min_index in peak_months:
                    score_right+=1
                    score_all+=1
                    print('Its a catch!')
                    print(score_right / score_all)
                else:
                    score_all += 1

    return score_right/score_all


# list to store files
filenames = []
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        filenames.append(path)
print("Filenames:")
print(filenames)

model_names = []
model_prefix = '100mil-scratch-2022-2022-'
model_numbers = ["%.2d" % i for i in range(1,13)]
for model_number in model_numbers:
    model_names.append(model_prefix+str(model_number))

models = []
for model_name in model_names:
    model = load_model(model_name)
    models.append(model)

cum_acc = 0
for file in filenames:
    cum_acc += score_file(file, models)

acc = cum_acc/len(filenames)
print("Final acc is: ")
print(acc)

