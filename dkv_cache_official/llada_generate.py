
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer
import time

import argparse

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cfg", type=float, default=0.0)

    parser.add_argument("--sampling-alg", type=str, default="low_confidence")
    parser.add_argument("--cache-steps", type=int, default=2)

    parser.add_argument("--origin", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--greedy", action="store_true")

    # hyper-parameter for q-cache
    parser.add_argument("--window-size", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def main():
    args = parser()

    device = 'cuda'
    if args.origin:
        from transformers import AutoModel as LLaDAModelLM
        from generation_utils.llada_generate import generate
    elif args.decode:
        from models.modeling_llada_dkv_cache_decode import LLaDAModelLM
        from generation_utils.llada_dkv_cache_decode import generate
    elif args.greedy:
        from models.modeling_llada_dkv_cache_greedy import LLaDAModelLM
        from generation_utils.llada_dkv_cache_greedy import generate
    else:
        raise NotImplementedError

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        #"John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have?",
    ] 

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [[{"role": "user", "content": "Please answer the question step by step and put the answer in \\boxed{}." + p}] for p in prompt]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    bsz = len(prompt)

    input_ids = tokenizer(
        prompt,
        padding_side = 'left',
        padding = 'longest'
    )['input_ids'] 
    input_ids = torch.tensor(input_ids).to(device)
   

    out = generate(
        model, tokenizer, input_ids, 
        steps=args.seq_len, gen_length=args.steps, block_length=args.block_size, 
        temperature=0., cfg_scale=0., 
        remasking=args.sampling_alg,
        enable_cache=not args.origin,
        cache_reloading_step=args.cache_steps,
        window_size=args.window_size
    )
    
    res = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for r in res:
        print('-' * 40)
        print(r, '\n')
        
    print('-' * 40)

    

if __name__ == '__main__':
    main()