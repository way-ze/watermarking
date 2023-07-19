## Attempting to implement Algorithm 2

# helper functions to perform likelihood ratio test
# TODO: Change code so that it runs via vector form! Do not use for loops for sum, speeds up the code a lot.
# TODO: add tqdm
# vocab_size = tokenizer.vocab_size

# Issues: Why does this go down infinitely??


import torch
import numpy as np
from bigram_estimator import pHat

def L_Gw(delta, w, word_dict, n_gram_dict, watermark_processor, tokenizer): 
    L = 0
    greenlist_w, redlist_w = watermark_processor._get_greenlist_ids(torch.tensor(tokenizer.encode(w)), get_redlist=True)
    greenlist_w = tokenizer.convert_ids_to_tokens(greenlist_w)
    redlist_w = tokenizer.convert_ids_to_tokens(redlist_w)
    
    for word in list(word_dict.keys())[:100]: # only go through first 100 words, for debugging purposes
        greensum = 0 
        redsum = 0
        for wdash in greenlist_w:
            green = pHat(wdash, word, word_dict, n_gram_dict)
            if green != 0: # prevents underflow
                # print("green = ", green)
                greensum += green
        for wdash in redlist_w:
            redsum += pHat(wdash, word, word_dict, n_gram_dict)
        if not (greensum == 0 and redsum == 0):
            L -= np.log(np.exp(delta) * greensum + redsum)
        if w in greenlist_w:
            L += delta
    return L, redsum