import torch
# import os
# import argparse
# from argparse import Namespace
from pprint import pprint
from functools import partial
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          OPTForCausalLM,
                          LogitsProcessorList)
# rom collections import Counter
# from nltk.util import ngrams 
# import json
from watermark_reliability_release.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

## modified code from kirchenbauer. Changed to generate a custom number of samples

def getGreenlist(w, watermark_processor, tokenizer=None):
    greenlist_w, redlist_w = watermark_processor._get_greenlist_ids(torch.tensor(tokenizer.encode(w)), get_redlist=True)
    greenlist_w = tokenizer.convert_ids_to_tokens(greenlist_w)
    redlist_w = tokenizer.convert_ids_to_tokens(redlist_w)
    return greenlist_w, redlist_w


def generate(prompt, args, model=None, device=None, tokenizer=None, times=1):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    #if args.prompt_max_length:
    #    pass
    #elif hasattr(model.config,"max_position_embedding"):
    #    args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    #else:
    #    args.prompt_max_length = 2048-args.max_new_tokens
    
    args.prompt_max_length = args.max_new_tokens
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    # print("Input IDs")
    # print(tokd_input["input_ids"][0][0])
    #print(watermark_processor._get_greenlist_ids(tokd_input["input_ids"]))
    # truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    # redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    #print(watermark_processor._get_greenlist_ids(redecoded_input))

    torch.manual_seed(args.generation_seed)
    
    corpus_without_watermark = []
    for i in range(times):
        # print(i)
        output_without_watermark = generate_without_watermark(**tokd_input)
        # print(output_without_watermark)
        corpus_without_watermark.append(tokenizer.decode(output_without_watermark[0], skip_special_tokens=False))
    
    # output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    
    corpus_with_watermark = []
    for i in range(times):
        output_with_watermark = generate_with_watermark(**tokd_input)
        corpus_with_watermark.append(tokenizer.decode(output_with_watermark[0], skip_special_tokens=False))
    
    
    # output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    # decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    # decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (tokd_input,
            watermark_processor,
            output_without_watermark,
            # redecoded_input,
            # int(truncation_warning),
            #decoded_output_without_watermark, 
            #decoded_output_with_watermark,
            #output_without_watermark,
            #output_with_watermark,
            corpus_without_watermark,
            corpus_with_watermark,
            args)

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
            # print("hi")
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer = tokenizer.to(device)

    return model, tokenizer, device