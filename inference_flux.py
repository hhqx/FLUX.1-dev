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
import csv
import json
import time
import torch
import torch_npu
import argparse
from FLUX1dev import FluxPipeline
from FLUX1dev import get_local_rank, get_world_size, initialize_torch_distributed

from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

cache_dict = {}
cache_dict['cache_start_block'] = 5
cache_dict['num_cache_layer_block'] = 13
cache_dict['cache_start_single_block'] = 1
cache_dict['num_cache_layer_single_block'] = 23
cache_dict['cache_start_steps'] = 18
cache_dict['cache_interval'] = 2

class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int = 1,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(max_num_prompts)
        else:
            print("This operation is not supported!")

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
    
    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, max_num_prompts: int):
        with open('hpsv2_benchmark_prompts.json', 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./flux", help="Path to the flux model directory")
    parser.add_argument("--save_path", type=str, default="./res", help="ouput image path")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", type=str, default="npu", help="NPU")
    parser.add_argument("--prompt_path", type=str, default="./prompts.txt", help="input prompt text path")
    parser.add_argument("--prompt_type", choices=["plain", "parti", "hpsv2"], default="plain", help="specify infer prompt type")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="specify image number every prompt generate")
    parser.add_argument("--max_num_prompt", type=int, default=0, help="limit the prompt number[0 indicates no limit]")
    parser.add_argument("--info_file_save_path", type=str, default="./image_info.json", help="path to save image info")
    parser.add_argument("--width", type=int, default=1024, help='Image size width')
    parser.add_argument("--height", type=int, default=1024, help='Image size height')
    parser.add_argument("--infer_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--use_cache", action="store_true", help="turn on dit cache or not")
    parser.add_argument("--device_type", choices=["A2-32g-single", "A2-32g-dual", "A2-64g"], default="A2-64g", help="specify device type")
    return parser.parse_args()

def infer(args):
    if args.device_type == "A2-32g-dual":
        from FLUX1dev import replace_tp_from_pretrain, replace_tp_extract_init_dict
        FluxPipeline.from_pretrained = classmethod(replace_tp_from_pretrain)
        FluxPipeline.extract_init_dict = classmethod(replace_tp_extract_init_dict)
    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16)
    if args.device_type == "A2-32g-single":
        torch.npu.set_device(args.device_id)
        pipe.enable_model_cpu_offload()
    elif args.device_type == "A2-64g":
        torch.npu.set_device(args.device_id)
        pipe.to(f"npu:{args.device_id}")
    elif args.device_type == "A2-32g-dual":
        local_rank = get_local_rank()
        world_size = get_world_size()
        initialize_torch_distributed(local_rank, world_size)
        pipe.to(f"npu:{local_rank}")
    else:
        raise ValueError(f"Not support device type:{args.device_type}")

    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, mode=0o640)

    infer_num = 0
    time_consume = 0
    current_prompt = None
    image_info = []
    prompt_loader = PromptLoader(args.prompt_path,
                                args.prompt_type,
                                1,
                                args.num_images_per_prompt,
                                args.max_num_prompt)
    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        save_names = input_info['save_names']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        print(f"[{infer_num+n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += 1
        if infer_num > 3:
            start_time = time.time()

        image = pipe(
            prompts,
            height=args.width,
            width=args.height,
            guidance_scale=3.5,
            num_inference_steps=args.infer_steps,
            max_sequence_length=512,
            use_cache=args.use_cache,
            cache_dict=cache_dict,
        ).images[0]

        if infer_num > 3:
            end_time = time.time() - start_time
            time_consume += end_time

        for j in range(n_prompts):
            image_save_path = os.path.join(args.save_path, f"{save_names[j]}.png")
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)
    
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)
    image_time_count = len(prompt_loader) - 3
    print(f"flux pipeline time is:{time_consume/image_time_count}")
    return


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)