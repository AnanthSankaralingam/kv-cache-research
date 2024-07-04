# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# dtore and compute attention by hand
from typing import List, Optional

import fire

from llama import Dialog, Llama, Attention
from llama.model import ModelArgs

import torch
import numpy as np 

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    prompt = "I am a college student and want to maximize my learning. Why is sleep important?"
    dialogs: List[Dialog] = [
        [{"role": "user", "content": prompt}],
    ]
    
    #modify generate method to return k and q by token, chat completion return kq, store kq in transformer

    results, xk, xq = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # count = 17
    # for key, value in xq.items():
    #   if count == 0:
    #     break
    #   count -=1
    #   print(key, value)

    Q_rows = []
    K_rows = []

    count = 17 #prompt length
    #todo: check if the first token weird here too. manually append it
    #todo: pad with 0s and create Q K. check if need to transpose K since alr transposed in model
    #todo: compute QK^T. print. visualize w heatmap
    for key, value in xq.items():
        if count == 0:
          break
        count -= 1
        Q_rows.append(value)
        # K_rows.append(xk[token])

    Q = torch.stack(Q_rows)
    # K = torch.stack(K_rows)

    # for dialog, result in zip(dialogs, results):
    #     # for msg in dialog:
    #     #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")



if __name__ == "__main__":
    fire.Fire(main)
