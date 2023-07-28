import torch
from bigram_estimator import pLM
from generate_watermark import getGreenlist
# import scipy


def L_Gw(delta, w, corpus, watermark_processor, tokenizer, model):
    ### ORIGINAL VERSION, not including the w_{t-1} filter 

    expdelta = torch.exp(delta)

    # gets greenlist of the word that we would like to use for the filter
    greenlist_w, redlist_w = getGreenlist(w, watermark_processor)

    # Tokenizes corpus into subwords
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
    tokenized_corpus = list(zip(*words_with_offsets))[0]
    corpus_t = tokenizer.encode(corpus, add_special_tokens=False)

    # Calculates softmaxed logits given the corpus as an input
    with torch.inference_mode():
        output = model(torch.tensor([corpus_t]))
    soft_logits = torch.softmax(output.logits[0], dim=1)

    # Initialise green and red sum
    green = torch.zeros(output.logits.size(1))
    red = torch.zeros(output.logits.size(1))

    # calculates p_LM(w'| w_{<t}) and sums over all words in green/redlist 
    for wdash in greenlist_w:
        green += pLM(wdash, tokenizer, soft_logits)
    for wdash in redlist_w:
        red += pLM(wdash, tokenizer, soft_logits)

    # calculates indicator function for $w_t \in G$, multiplies by delta
    deltas = torch.zeros(output.logits.size(1))
    for i in range(output.logits.size(1)):
        # added log(p_LM(w_t | w_{<t}))
        deltas[i] = deltas[i] + torch.log(pLM(tokenized_corpus[i], tokenizer, soft_logits))
        if tokenized_corpus[i] in greenlist_w:
            deltas[i] = deltas[i] + delta
    
    # sum everything up, takes logarithm
    Lvector = deltas - torch.log(expdelta * green + red)
    L_final = torch.sum(Lvector)

    return L_final


def L_Gw2(delta, w, corpus, watermark_processor, tokenizer, model):
    ### NEW VERSION, including the w_{t-1} filter as discussed with Anej on 28.7.23 
    expdelta = torch.exp(delta)

    # gets greenlist of the word that we would like to use for the filter
    greenlist_w, redlist_w = getGreenlist(w, watermark_processor)

    # Tokenizes corpus into subwords
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
    tokenized_corpus = list(zip(*words_with_offsets))[0]
    corpus_t = tokenizer.encode(corpus, add_special_tokens=False)


    # Calculates softmaxed logits given the corpus as an input
    with torch.inference_mode():
        output = model(torch.tensor([corpus_t]))
    soft_logits = torch.softmax(output.logits[0], dim=1)

    # NEW: filters out all w_{t-1} = w
    indices = [i for i, x in enumerate(tokenized_corpus) if x == w]
    # gets indices of w_t
    indices = [i + 1 for i in indices]
    soft_logits_filtered = soft_logits[indices,:]

    # Initialise green and red sum
    green = torch.zeros(len(indices))
    red = torch.zeros(len(indices))
    
    # Same stuff as above
    for wdash in greenlist_w:
        green += pLM(wdash, tokenizer, soft_logits_filtered)
    for wdash in redlist_w:
        red += pLM(wdash, tokenizer, soft_logits_filtered)
    
    deltas = torch.zeros(len(indices))
    for i in range(len(indices)):
        if tokenized_corpus[indices[i]] in greenlist_w:
            deltas[i] = deltas[i] + delta
    Lvector = deltas - torch.log(expdelta * green + red)

    L_final = torch.sum(Lvector)
    return L_final

def likelihoodRatioTest(statistic1, statistic2):
    raise NotImplementedError
    return