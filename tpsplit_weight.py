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
import shutil
import torch
import argparse
from safetensors.torch import load_file, save_file

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./flux", help="Path to the flux model directory")
    return parser.parse_args()
    #TODO: support multicard > 2 split

def split_weight(file_path, transformer_path_0, transformer_path_1):
    init_dict = load_file(file_path)
    file_name = file_path.split('/')[-1]
    
    dict_rank0 = {}
    dict_rank1 = {}
    for key in init_dict:
        #for MLP module:
        if 'ff' in key:
            if 'net.0' in key and 'weight' in key:  #col split
                shape = init_dict[key].shape[0]
                dict_rank0[key] = init_dict[key][:shape//2,].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:,].contiguous()
            elif 'net.0' in key and 'bias' in key:  #col split
                shape = init_dict[key].shape[0]
                dict_rank0[key] = init_dict[key][:shape//2].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:].contiguous()
            elif 'net.2' in key and 'weight' in key:    #row split
                shape = init_dict[key].shape[1]
                dict_rank0[key] = init_dict[key][..., :shape//2].contiguous()
                dict_rank1[key] = init_dict[key][..., shape//2:].contiguous()
            elif 'net.2' in key and 'bias' in key:    #only card 0 support
                dict_rank0[key] = init_dict[key].contiguous()
            else:
                dict_rank0[key] = init_dict[key].contiguous()
                dict_rank1[key] = init_dict[key].contiguous()
        #for FA module:
        elif 'attn' in key:
            if 'norm' in key:
                dict_rank0[key] = init_dict[key].contiguous()
                dict_rank1[key] = init_dict[key].contiguous()
            elif 'out' in key and 'weight' in key:
                shape = init_dict[key].shape[1]
                dict_rank0[key] = init_dict[key][..., :shape//2].contiguous()
                dict_rank1[key] = init_dict[key][..., shape//2:].contiguous()
            elif 'out' in key and 'bias' in key:
                dict_rank0[key] = init_dict[key].contiguous()
            elif 'weight' in key:           #qkv linear
                shape = init_dict[key].shape[0]
                dict_rank0[key] = init_dict[key][:shape//2,].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:,].contiguous()
            elif 'bias' in key:
                shape = init_dict[key].shape[0]
                dict_rank0[key] = init_dict[key][:shape//2].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:].contiguous()
            else:
                dict_rank0[key] = init_dict[key].contiguous()
                dict_rank1[key] = init_dict[key].contiguous()
        #for linear before and after single_block fa:
        elif 'single_transformer_blocks' in key and 'proj' in key:
            shape = init_dict[key].shape[0]
            if 'weight' in key:
                dict_rank0[key] = init_dict[key][:shape//2,].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:,].contiguous()
            elif 'bias' in key:
                dict_rank0[key] = init_dict[key][:shape//2].contiguous()
                dict_rank1[key] = init_dict[key][shape//2:].contiguous()
        else:
            dict_rank0[key] = init_dict[key].contiguous()
            dict_rank1[key] = init_dict[key].contiguous()
    
    save_file(dict_rank0, os.path.join(transformer_path_0, file_name))
    save_file(dict_rank1, os.path.join(transformer_path_1, file_name))


if __name__ == "__main__":
    args = parse_arguments()

    transformer_path = os.path.join(args.path, 'transformer')
    if not os.path.exists(transformer_path):
        print(f"the model path:{args.path} does not contain transformer, please check")
        raise ValueError
    
    transformer_path_0 = transformer_path + '_0'
    if not os.path.exists(transformer_path_0):
        os.makedirs(transformer_path_0, mode=0o640)
    transformer_path_1 = transformer_path + '_1'
    if not os.path.exists(transformer_path_1):
        os.makedirs(transformer_path_1, mode=0o640)

    for file in os.listdir(transformer_path):
        file_type = file.split('.')[-1]
        file_path = os.path.join(transformer_path, file)
        if file_type != 'safetensors':
            shutil.copy(file_path, transformer_path_0)
            shutil.copy(file_path, transformer_path_1)
        else:
            split_weight(file_path, transformer_path_0, transformer_path_1)