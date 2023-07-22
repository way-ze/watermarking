## Attempting to implement Algorithm 2

# helper functions to perform likelihood ratio test
# TODO: Change code so that it runs via vector form! Do not use for loops for sum, speeds up the code a lot.
# TODO: add tqdm
# vocab_size = tokenizer.vocab_size


import torch
from bigram_estimator import pLM

def L_Gw(delta, w, word_dict, watermark_processor, tokenizer, model): 
    L = 0
    greenlist_w, redlist_w = watermark_processor._get_greenlist_ids(torch.tensor(tokenizer.encode(w)), get_redlist=True)
    greenlist_w = tokenizer.convert_ids_to_tokens(greenlist_w)
    redlist_w = tokenizer.convert_ids_to_tokens(redlist_w)
    
    for word in list(word_dict.keys())[:10]:
        # print(f"word = {word}")
        greensum = 0 
        redsum = 0
        
        for wdash in greenlist_w[:100]:
            # print(f"wdash_green = {wdash}")
            # green = pHat(wdash, word, word_dict, n_gram_dict)
            green = pLM(wdash, word, tokenizer, model)
            if green != 0: # prevents underflow
                greensum += green

        for wdash in redlist_w[:100]:
            # print(f"wdash_red = {wdash}")
            red = pLM(wdash, word, tokenizer, model)
            if red != 0:
                redsum += red
        
        if not (greensum == 0 and redsum == 0):
            L -= torch.log(torch.exp(delta) * greensum + redsum)
        if w in greenlist_w:
            L += delta
    
        print(f"word = {word}, greensum = {greensum}, redsum = {redsum}, sum = {greensum + redsum}")
        # greensum + redsum has to be 1, intuitively??
        # can we not just force redsum to be 1 - greensum?
    return L, redsum