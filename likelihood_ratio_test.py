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
    
    for word in list(word_dict.keys())[1]:
        greensum = 0 
        redsum = 0
        word2tok = tokenizer.convert_tokens_to_ids(word)
        with torch.inference_mode():
            output = model(torch.tensor([[word2tok]]))
        for wdash in greenlist_w:
            green = pLM(wdash, word, tokenizer, output)
            if green != 0: # prevents underflow
                greensum += green

        for wdash in redlist_w:
            red = pLM(wdash, word, tokenizer, output)
            if red != 0:
                redsum += red
        
        if not (greensum == 0 and redsum == 0):
            L -= torch.log(torch.exp(delta) * greensum + redsum)
        if w in greenlist_w:
            L += delta
    
        print(f"word = {word}, greensum = {greensum}, redsum = {redsum}, sum = {greensum + redsum}")

    return L, redsum