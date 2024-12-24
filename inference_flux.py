#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import torch
import torch_npu
import argparse
from .FLUX1dev import FluxPipeline

from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            batch_size: int = 1,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        self.load_prompts(prompt_file, max_num_prompts)

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration
        
        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret
    
    def load_prompts(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./flux", help="Path to the flux model directory")
    parser.add_argument("--save_path", type=str, default="./res", help="ouput image path")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", type=str, default="npu", help="NPU")
    parser.add_argument("--prompt_path", type=str, default="./promts.txt", help="input prompt text path")
    parser.add_argument("--input_size", type=tuple, default=(1024, 1024), help='Image size (h, w)')
    parser.add_argument("--infer_steps", type=int, default=25, help="Inference steps")
    parser.add_argument('--seed', type=int, default=42, help="A seed for all the prompts")
    return parser.parse_args()

def infer(args):
    torch.npu.set_device(args.device_id)
    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, mode=0o640)

    infer_num = 0
    time_consume = 0
    prompt_loader = PromptLoader(args.prompt_path)
    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        save_names = input_info['save_names']

        print(f"[{infer_num}/{len(prompt_loader)}]: {prompts}")
        infer_num += 1
        if infer_num > 3:
            start_time = time.time()

        image = pipe(
            prompts,
            height=args.input_size[0],
            width=args.input_size[1],
            guidance_scale=3.5,
            num_inference_steps=args.infer_steps,
            max_sequence_length=512,
            generator=torch.Generator().manual_seed(args.seed)
        ).images[0]

        if infer_num > 3:
            end_time = time.time() - start_time
            time_consume += end_time
        image_save_path = os.path.join(args.save_path,f"{save_names[0]}.png")
        image.save(image_save_path)
    
    image_time_count = len(prompt_loader) - 3
    print(f"flux pipeline time is:{time_consume/image_time_count}")
    return


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)