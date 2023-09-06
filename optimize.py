import os
import json

from argparse import Namespace 
import matplotlib.pyplot as plt 
import numpy as np
import torch


# from bigram_estimator import pLM
from generate_watermark import load_model, generate
from likelihood_ratio_test import L_Gw2

directory_wm = 'data/wmed'
 

corpus_with_watermark = ""
for filename in os.listdir(directory_wm):
    f = os.path.join(directory_wm, filename)
    with open(f, 'r') as infile:
        corpus_with_watermark += json.loads(infile.read())
        
directory_no_wm = 'data/no-wmed'

corpus_without_watermark = ""
for filename in os.listdir(directory_no_wm):
    f = os.path.join(directory_no_wm, filename)
    with open(f, 'r') as infile:
        corpus_without_watermark += json.loads(infile.read())
        
        args_test = Namespace()

arg_dict_test = {
    'run_gradio': False, 
    'demo_public': False,
    'model_name_or_path': 'facebook/opt-350m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 200, 
    'generation_seed': 456, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 1, # 0.7, # we want to sample exactly from the wm-ed distribution
    
    # 'use_gpu': False, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.25, 
    'delta': 2, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
}

args_test.__dict__.update(arg_dict_test)

args_test.is_seq2seq_model = any([(model_type in args_test.model_name_or_path) for model_type in ["t5","T0"]])
args_test.is_decoder_only_model = any([(model_type in args_test.model_name_or_path) for model_type in ["gpt","opt","bloom"]])


model_2, tokenizer_2, device_2 = load_model(args_test)

times = 1 #1000
tokd_input, watermark_processor, output, _, _, _ = generate("",
                                                                args_test,
                                                                model=model_2,
                                                                device=device_2,
                                                                tokenizer=tokenizer_2,
                                                                times=times)

delta_init = torch.tensor(1.0)
delta_init.requires_grad = True
loss = torch.tensor(1.0)
loss.requires_grad = True
tol = 1e-9
deltaarr = torch.linspace(0, 1, 50)

w = 'Ġthe'
optimizer = torch.optim.SGD([delta_init], lr=0.01) # use normal gd

epochs = 3

def L_Gw_delta(delta):
  return L_Gw2(delta, w, corpus_with_watermark, watermark_processor, tokenizer_2, model_2, device_2)

for _ in range(epochs):
  optimizer.zero_grad()
  loss = -L_Gw_delta(delta_init)
  loss.backward()
  optimizer.step()
  print(f"Loss = {loss.item()}, delta_grad = {optimizer.param_groups[0]['params'][0].grad}, delta = {delta_init.item()}")


print(f'found optimal value of delta={delta_init.item()}, with LGWDelta(delta)={L_Gw_delta(delta_init).item()}')