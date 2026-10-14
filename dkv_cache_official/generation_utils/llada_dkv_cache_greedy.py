import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer
from models.modeling_llada_dkv_cache_greedy import LLaDAModelLM

import time

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def find_window_tokens(window_index, select_ids, mask_id, window_size):
    '''
    Find the window tokens in the input tensor x.
    '''
    for select_id in select_ids:
        left_ptr, right_ptr = select_id - 1, select_id + 1
        cnt = 0
        left_flag = True
        while cnt < window_size:
            if left_flag:
                if left_ptr >= 0 and not window_index[left_ptr] and mask_id[left_ptr]:
                    window_index[left_ptr] = True
                    cnt += 1
                left_ptr -= 1
            else:
                if right_ptr < window_index.shape[0] and not window_index[right_ptr] and mask_id[right_ptr]:
                    window_index[right_ptr] = True
                    cnt += 1
                right_ptr += 1
            left_flag = not left_flag

            if left_ptr < 0 and right_ptr > window_index.shape[0]:
                break

    return window_index


@ torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, enable_cache=False, cache_reloading_step=1, window_size=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    B, L = prompt.shape
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        past_qkv = None
        prv_transfer_idx, cur_transfer_index = None, None

        B = prompt.shape[0]

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        accumlated_num_transfer_tokens = torch.cumsum(
            torch.cat(
                [torch.zeros((B, 1), device=num_transfer_tokens.device, dtype=num_transfer_tokens.dtype), num_transfer_tokens], dim=-1
            ),
            dim=-1
        )
        
        if remasking == 'low_confidence':
            raise NotImplementedError("Not support for low confidence")
        elif remasking == 'random':
            x0_p = torch.rand((x.shape[0], x.shape[1]), device=x.device)
        else:
            raise NotImplementedError(remasking)

        x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
        x0_p[:, :prompt.shape[1] + num_block * block_length] = -np.inf

        # rank the order in x0_p
        denoising_index = torch.sort(x0_p, descending=True).indices
        #print("denoising order = ", denoising_index[:, :block_length])

        start_time = time.time()
        for i in range(steps):
            #print(f"Step {i}:")

            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                raise NotImplementedError('cfg_scale > 0 is not supported yet for cache.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if i % cache_reloading_step != 0 and enable_cache:
                    model.set_qkv_cache(True, True)

                    model.get_pos_rotary_embedding_cache(cur_transfer_index)
                    next_x = x[cur_transfer_index].view(x.shape[0], -1)
                    #print("Input: ", next_x.shape)
                    outputs = model(next_x, 
                        past_query_key_values = past_qkv, 
                        use_cache = True, preprocess_cache = True
                    )
                    
                    model.set_qkv_cache(False, False)
                    model.reset_pos_rotary_embedding_cache()
                    
                elif enable_cache:
                    #print("Input: ", x.shape)

                    outputs = model(
                        x, past_query_key_values = past_qkv, 
                        use_cache = False, preprocess_cache = True
                    )
                else:
                    outputs = model(
                        x, past_query_key_values = None, 
                        use_cache = False, preprocess_cache = False
                    )
                    
                logits = outputs.logits
                past_qkv = outputs.past_key_values

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature) # b, l, D
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if x0.shape[1] < x.shape[1]: # for cache, refill x0 to be the original order and size
                reorder_token_idx = torch.nonzero(cur_transfer_index, as_tuple=True)[1].view(x0_p.shape[0], -1)
                refill_x0 = torch.full((x.shape[0], x.shape[1]), mask_id, device=x0.device, dtype=x0.dtype)
                refill_x0 = refill_x0.scatter_(1, reorder_token_idx, x0)
                x0 = refill_x0
                
            x0 = torch.where(mask_index, x0, x)
        
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            next_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(x0.shape[0]):
                select_index = denoising_index[j, accumlated_num_transfer_tokens[j, i]:accumlated_num_transfer_tokens[j, i+1]]
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index] 

            for j in range(x0.shape[0]):
                select_index = denoising_index[j, accumlated_num_transfer_tokens[j, i]:accumlated_num_transfer_tokens[j, i+1]]

                if i < steps-1:
                    next_select_index = denoising_index[j, accumlated_num_transfer_tokens[j, i+1]:accumlated_num_transfer_tokens[j, i+1] + num_transfer_tokens[j, i+1]]
                    next_transfer_index[j, next_select_index] = True

                    # For decoded token
                    next_transfer_index[j, select_index] = True

                    # for window token
                    if window_size > 0:
                        ori = False
                        if ori:
                            find_window_tokens(next_transfer_index[j], select_index, mask_index[j], window_size=window_size)
                        else:
                            find_window_tokens(next_transfer_index[j], next_select_index, mask_index[j], window_size=window_size)
            
            #all_transfer_index = (x != mask_id)
            # TODO: check if prv-transfer-idx and cur-transfer-idx work at the final order of the sequence
            prv_transfer_idx, cur_transfer_index = cur_transfer_index, next_transfer_index
            past_qkv = [past_qkv, (prv_transfer_idx, cur_transfer_index)]

            
            #if prv_transfer_idx is not None:
            #    print("Prv transfer:", prv_transfer_idx.nonzero(as_tuple=True)[1].view(B, -1))
            
            #if cur_transfer_index is not None:
            #    print("Cur transfer:", cur_transfer_index.nonzero(as_tuple=True)[1].view(B, -1))

            #if next_transfer_index is not None:
            #    print("Next transfer:", next_transfer_index.nonzero(as_tuple=True)[1].view(B, -1))
            #print(x[:, prompt.shape[1]:]) # For Debug
            #for i in range(B):
            #print("Step {}: {}".format(i, tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]))
        
    return x