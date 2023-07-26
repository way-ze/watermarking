## Attempting to implement Algorithm 2

# helper functions to perform likelihood ratio test
# TODO: Change code so that it runs via vector form! Do not use for loops for sum, speeds up the code a lot.
# TODO: add tqdm
# vocab_size = tokenizer.vocab_size


import torch
from bigram_estimator import pLM
import scipy

def L_Gw(delta, w, corpus, watermark_processor, tokenizer, model):
    expdelta = torch.exp(delta)
    greenlist_w, redlist_w = watermark_processor._get_greenlist_ids(torch.tensor(tokenizer.encode(w)), get_redlist=True)
    greenlist_w = tokenizer.convert_ids_to_tokens(greenlist_w)
    redlist_w = tokenizer.convert_ids_to_tokens(redlist_w)

    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
    tokenized_corpus = list(zip(*words_with_offsets))[0]
    
    corpus_t = tokenizer.encode(corpus, add_special_tokens=False)

    with torch.inference_mode():
        output = model(torch.tensor([corpus_t]))
    soft_logits = torch.softmax(output.logits[0], dim=1)

    green = torch.zeros(output.logits.size(1))
    red = torch.zeros(output.logits.size(1))

    for wdash in greenlist_w:
        green += pLM(wdash, tokenizer, soft_logits)

    for wdash in redlist_w:
        red += pLM(wdash, tokenizer, soft_logits)

    deltas = torch.zeros(output.logits.size(1))
    for i in range(output.logits.size(1)):
        if tokenized_corpus[i] in greenlist_w:
            deltas[i] = deltas[i] + delta

    L = torch.sum(deltas - torch.log(expdelta * green + red))
    return L

# def L_Gw(delta, w, word_dict, watermark_processor, tokenizer, model): 
#     Lgrprod = 1
#     Ldelta = 0
#     expdelta = torch.exp(delta)
#     greenlist_w, redlist_w = watermark_processor._get_greenlist_ids(torch.tensor(tokenizer.encode(w)), get_redlist=True)
#     greenlist_w = tokenizer.convert_ids_to_tokens(greenlist_w)
#     redlist_w = tokenizer.convert_ids_to_tokens(redlist_w)
    
#     for word in word_dict.keys():
#         greensum = 0 
#         redsum = 0
#         word2tok = tokenizer.convert_tokens_to_ids(word)
#         with torch.inference_mode():
#             output = model(torch.tensor([[word2tok]]))
#         soft_next_token_logits = torch.softmax(output.logits[0, -1, :], -1) 
        
#         for wdash in greenlist_w:
#             greensum += pLM(wdash, tokenizer, soft_next_token_logits)

#         for wdash in redlist_w:
#             redsum += pLM(wdash, tokenizer, soft_next_token_logits)
        
#         Lgrprod *= greensum * expdelta + redsum
#         if w in greenlist_w:
#             Ldelta += delta
    
#     L = Ldelta - torch.log(Lgrprod)
#     return L

def likelihoodRatioTest(statistic1, statistic2):
    raise NotImplementedError
    return