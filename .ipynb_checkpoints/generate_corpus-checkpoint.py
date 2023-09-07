from generate_watermark import generate, load_model
import json
import os
from argparse import Namespace

def main(args): 
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        input_text = (
        ""
        )

        args.default_prompt = input_text

        term_width = 80
        print("#"*term_width)
        print("Prompt:")
        print(input_text)
        times = 1000
        _, _, _, corpus_without_watermark, corpus_with_watermark, _ = generate("",
                                                                    args,
                                                                    model=model,
                                                                    device=device,
                                                                    tokenizer=tokenizer,
                                                                    times=times)
        
        with open("corpus_with_watermark_407.json", "w") as file:
            json.dump(corpus_with_watermark, file)

        with open("corpus_without_watermark_407.json", "w") as file:
            json.dump(corpus_without_watermark, file)
        
    return

arg_dict = {
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
    'generation_seed': 408, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 1, # 0.7, # we want to sample exactly from the wm-ed distribution
    
    'use_gpu': True, 
    # 'use_gpu': False, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.25, 
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
}

args = Namespace()
args.__dict__.update(arg_dict)

args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])

main(args)