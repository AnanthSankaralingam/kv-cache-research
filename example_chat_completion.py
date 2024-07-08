# script to generate k and q matrixes to compute attention based on custom prompt

from typing import List, Optional

import fire

from llama import Dialog, Llama, Attention
from llama.model import ModelArgs

import torch
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

def main(
    ckpt_dir: str = "/content/llama3/Meta-Llama-3-8B",
    tokenizer_path: str = "/content/llama3/Meta-Llama-3-8B/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
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

    #print out relevant info about each key/query
    # count = 1
    # for key, value in xq.items():
    #   if count == 0:
    #     break
    #   count -=1
    #   print(key + ", query shape: ", value)

    #store Q and K tensors for all tokens. each index is a tensor, then combine for big matrixes
    Q_rows = []
    K_rows = []

    #TODO: check if the first token weird here too. manually append it

    count = 17 #prompt length
    #append query values to array, skip first for mismatch size
    for key, value in xq.items():
        if count == 0:
          break
        count -= 1
        if count == 16:
          continue
        Q_rows.append(value)

    count = 17 #prompt length
    #append key values to array, skip first for mismatch size
    for key, value in xk.items():
        if count == 0:
          break
        count -= 1
        if count == 16:
          continue
        K_rows.append(value)

    Q = torch.stack(Q_rows)
    K = torch.stack(K_rows)

    print("K shape:", K.shape)
    print("Q shape:", Q.shape)
    # QK^T
    K_T = K.transpose(-1, -2)  # Transpose the last two dimensions
    att = torch.matmul(Q, K_T)

    print("Attention shape:", att.shape)
    # print(att)

    # TODO: visualization
    att_2d = att.mean(0).squeeze(0).squeeze(-1).cpu().float().numpy()

    # Visualize attention with heatmap - FIX
    sns.set()
    plt.figure(figsize=(8, 8))
    sns.heatmap(att_2d, cmap="YlGnBu", square=True)
    plt.show(block=True)


    # for dialog, result in zip(dialogs, results):
    #     # for msg in dialog:
    #     #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")



if __name__ == "__main__":
    fire.Fire(main)
