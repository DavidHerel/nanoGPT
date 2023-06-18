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
# Importing Pandas to create DataFrame
import pandas as pd



init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
starts_with = "\n " # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# dtype='float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
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

    start_ids = encode(starts_with+start)
    x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...])
    y = (torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            logits, loss = model(x, y)
            return loss.exp().item()


def score_file(filename, models, df):
    score_right = 0
    score_all = 0
    # initializing the titles and rows list
    rows = []
    new_row = {}

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
        # print("stability: "+str(stability))
        peak_months=[]
        new_row['stability'] = '0'
        new_row['peak_months'] = []
        if stability=='0':
            peak_months=ast.literal_eval(row[6])
            new_row['stability'] = '1'
            new_row['peak_months']=peak_months


        print("peak_months: ")
        print(peak_months)

        # parsing each column of a row
        for col in row[:5]:
            col = col.lower()
            print(col)
            new_row['sentence']=col
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
                new_row['perplexities'] = ppls
                new_row['rstd'] = rstd
                #if rstd <0.25
                if rstd < 0.25:
                    score_right+=1
                    score_all+=1
                    new_row['success'] = '1'
                    print('rstd:' +str(rstd))
                    print('Its a catch!')
                    print(score_right / score_all)
                else:
                    score_all+=1
                    new_row['success'] = '0'
            else:
                ppls = []
                for model in models:
                    ppl = get_ppl(model, col)
                    print(ppl)
                    ppls.append(ppl)
                new_row['perplexities'] = ppls
                #get minimum index
                min_index = ppls.index(min(ppls))+1
                print("our_peak_month: "+str(min_index))
                new_row['our_peak_month'] = min_index
                if min_index in peak_months:
                    score_right+=1
                    score_all+=1
                    print('Its a catch!')
                    print(score_right / score_all)
                    new_row['success'] = '1'
                else:
                    score_all += 1
                    new_row['success'] = '0'
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print()

    return score_right/score_all, df


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
size_prefix='1bil'
model_prefix = size_prefix+'-scratch-2022-2022-'
model_numbers = ["%.2d" % i for i in range(1,13)]
for model_number in model_numbers:
    model_names.append(model_prefix+str(model_number))

models = []
for model_name in model_names:
    model = load_model(model_name)
    models.append(model)

cum_acc = 0
# dictionary with list object in values
all_dict = {
}

for file in filenames:
    # Creating Empty DataFrame and Storing it in variable df - export it
    df = pd.DataFrame(columns=['sentence', 'stability', 'rstd','perplexities','peak_months','our_peak_month', 'success'])
    acc, df = score_file(file, models, df)
    cum_acc+=acc
    df.to_csv(size_prefix+file, index=False)
    print(file+': '+str(acc))

    all_dict[file]=acc

acc = cum_acc/len(filenames)
all_dict['all']=acc
print("Final acc is: ")
print(acc)

# creating a Dataframe object
df = pd.DataFrame.from_dict(all_dict)
df.to_csv(size_prefix, index=False)

