#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
# limitations under the License


import os
from functools import reduce

MAX_PATH_LENGTH = 4096
MAX_FILENUM_PER_DIR = 1024
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024
SAFEOPEN_FILE_PERMISSION = 0o640
MAX_PROMPT_LENGTH = 1024

FLAG_OS_MAP = {
    'r': os.O_RDONLY, 'r+': os.O_RDWR,
    'w': os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
    'w+': os.O_CREAT | os.O_TRUNC | os.O_RDWR,
    'a': os.O_CREAT | os.O_APPEND | os.O_WRONLY,
    'a+': os.O_CREAT | os.O_APPEND | os.O_RDWR,
    'x': os.O_CREAT | os.O_EXCL,
    "b": getattr(os, "O_BINARY", 0)
}


def safe_open(file_path: str, mode='r', encoding=None, permission_mode=0o640, **kwargs):
    """
    Args:
        file_path (str): 文件路径
        mode (str): 文件打开模式
        encoding (str): 文件编码方式
        permission_mode: 文件权限最大值
        max_path_length (int): 文件路径最大长度
        max_file_size (int): 文件最大大小，单位: 字节, 默认值10MB
        check_link (bool): 是否校验软链接
        kwargs:
    """
    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_FILE_SIZE)
    check_link = kwargs.get('check_link', True)

    file_path = standardize_path(file_path, max_path_length, check_link)
    check_file_safety(file_path, max_file_size, permission_mode)

    flags = []
    for item in list(mode):
        if item == "+" and flags:
            flags[-1] = f"{flags[-1]}+"
            continue
        flags.append(item)
    flags = [FLAG_OS_MAP.get(mode, os.O_RDONLY) for mode in flags]
    total_flag = reduce(lambda a, b: a | b, flags)

    return os.fdopen(os.open(file_path, total_flag, SAFEOPEN_FILE_PERMISSION),
                     mode, encoding=encoding)


def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    Check and standardize path.
    Args:
        path (str): 未标准化路径
        max_path_length (int): 文件路径最大长度
        check_link (bool): 是否校验软链接
    Return: 
        path (str): 标准化后的绝对路径
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path


def is_path_exists(path: str):
    return os.path.exists(path)


def check_path_is_none(path: str):
    if path is None:
        raise ValueError("The path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise ValueError(f"The path:{path} is a symbolic link file.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    if path.__len__() > max_path_length:
        raise ValueError(f"The length of path is {path.__len__()}, which exceeds the limit {max_path_length}.")


def check_file_size_lt(path: str, max_file_size=MAX_FILE_SIZE):
    if os.path.getsize(path) > max_file_size:
        raise ValueError(
            f"The size of file:{path} is {os.path.getsize(path)}, which exceeds the limit {max_file_size}.")


def check_owner(path: str):
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    if not (os.geteuid() == 0 or path_gid in os.getgroups() or user_check):
        raise ValueError(f"The path:{path} is not owned by current user or root")
    

def check_max_permission(file_path: str, permission_mode=0o640):
    # check permission
    file_mode = os.stat(file_path).st_mode & 0o777 # use 777 as mask to get 3-digit octal number
    # transeform file_mode into binary patten,remove the head '0b' string,expand to 9 bits
    file_mode_bin = bin(file_mode)[2:].zfill(9)
    # transeform permission_mode into binary patten,remove the head '0b' string,expand to 9 bits
    max_mode_bin = bin(permission_mode)[2:].zfill(9)
    for i in range(9): # 9 means 9-bit binary number, checking every bit
        if file_mode_bin[i] > max_mode_bin[i]:
            raise ValueError(f'The permission of {file_path} is higher than {oct(permission_mode)}')


def check_file_safety(file_path: str, max_file_size=MAX_FILE_SIZE, is_check_file_size=True, permission_mode=0o640):
    if not is_path_exists(file_path):
        raise ValueError(f"The path:{file_path} doesn't exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"The input:{file_path} is not a file.")
    if is_check_file_size:
        check_file_size_lt(file_path, max_file_size)
    check_owner(file_path)
    check_max_permission(file_path, permission_mode)


def check_file_num_lt(path: str, max_file_num=MAX_FILENUM_PER_DIR):
    filenames = os.listdir(path)
    if len(filenames) > max_file_num:
        raise ValueError(
            f"The file num in dir{path} is {len(filenames)}, which exceeds the limit {max_file_num}."
        )


def check_dir_safety(dir_path: str, max_file_num=MAX_FILENUM_PER_DIR, is_check_file_num=True, permission_mode=0o750):
    if not is_path_exists(dir_path):
        raise ValueError(f"the path{dir_path} does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"the path{dir_path} is not a dir.")
    if is_check_file_num:
        check_file_num_lt(dir_path, max_file_num)
    check_owner(dir_path)
    check_max_permission(dir_path, permission_mode)


def check_param_valid(height, width, infer_step):
    if height <= 0:
        raise ValueError(f"Param height invalid, expected positive value, but get {height}")
    if width <= 0:
        raise ValueError(f"Param width invalid, expected positive value, but get {width}")
    if infer_step <= 0:
        raise ValueError(f"Param infer_step invalid, expected positive value, but get {infer_step}")



def check_prompts_valid(prompts):
    if type(prompts) is list:
        for prompt in prompts:
            if len(prompt) == 0 or len(prompt) >= MAX_PROMPT_LENGTH:
                raise ValueError(
                    f"The length of the prompt should be (0, {MAX_PROMPT_LENGTH}), 
                        but prompts:{prompt} length is {len(prompt)}.")
    elif type(prompts) is str:
        if len(prompts) == 0 or len(prompts) >= MAX_PROMPT_LENGTH:
            raise ValueError(
                f"The length of the prompt should be (0, {MAX_PROMPT_LENGTH}), 
                    but prompts:{prompts} length is {len(prompts[0])}.")