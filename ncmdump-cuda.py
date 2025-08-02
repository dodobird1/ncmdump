# -*- coding: utf-8 -*-
__author__ = 'qyh feat. dodobird1'
__date__ = '2025/07/30 10:00'

import binascii
import struct
import base64
import json
import os
from Crypto.Cipher import AES
import argparse
import glob
import sys

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    HAVE_PYCUDA = True
except ImportError:
    HAVE_PYCUDA = False

# CUDA解密内核（如果可用）
if HAVE_PYCUDA:
    kernel_code = """
    __global__ void decrypt_kernel(unsigned char *data, unsigned char *key_box, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            unsigned char j = idx & 0xff;
            unsigned char a = key_box[j];
            unsigned char b = key_box[(a + j) & 0xff];
            data[idx] ^= key_box[(a + b) & 0xff];
        }
    }
    """
    mod = SourceModule(kernel_code)
    cuda_decrypt_kernel = mod.get_function("decrypt_kernel")

def gpu_decrypt(data, key_box):
    """使用CUDA GPU解密数据"""
    n = len(data)
    # 分配设备内存
    data_gpu = cuda.mem_alloc(n)
    key_box_gpu = cuda.mem_alloc(256)
    
    # 复制数据到设备
    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(key_box_gpu, key_box)
    
    # 计算网格和块大小
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # 执行内核
    cuda_decrypt_kernel(data_gpu, key_box_gpu, n,
                        block=(threads_per_block, 1, 1),
                        grid=(blocks_per_grid, 1))
    
    # 将结果复制回主机
    result = bytearray(n)
    cuda.memcpy_dtoh(result, data_gpu)
    return bytes(result)

def cpu_decrypt(data, key_box):
    """使用CPU解密数据"""
    data_array = bytearray(data)
    n = len(data_array)
    for i in range(1, n + 1):
        j = i & 0xff
        a = key_box[j]
        b = key_box[(a + j) & 0xff]
        data_array[i - 1] ^= key_box[(a + b) & 0xff]
    return bytes(data_array)

def dump(file_path, output_dir=None, del_ncm=False, use_gpu=True, not_overwrite=False):
    core_key = binascii.a2b_hex("687A4852416D736F356B496E62617857")
    meta_key = binascii.a2b_hex("2331346C6A6B5F215C5D2630553C2728")
    unpad = lambda s: s[0:-(s[-1] if type(s[-1]) == int else ord(s[-1]))]
    
    with open(file_path, 'rb') as f:
        header = f.read(8)
        assert binascii.b2a_hex(header) == b'4354454e4644414d'
        f.seek(2, 1)
        key_length = f.read(4)
        key_length = struct.unpack('<I', bytes(key_length))[0]
        key_data = f.read(key_length)
        key_data_array = bytearray(key_data)
        for i in range(0, len(key_data_array)):
            key_data_array[i] ^= 0x64
        key_data = bytes(key_data_array)
        cryptor = AES.new(core_key, AES.MODE_ECB)
        key_data = unpad(cryptor.decrypt(key_data))[17:]
        key_length = len(key_data)
        key_data = bytearray(key_data)
        key_box = bytearray(range(256))
        c = 0
        last_byte = 0
        key_offset = 0
        for i in range(256):
            swap = key_box[i]
            c = (swap + last_byte + key_data[key_offset]) & 0xff
            key_offset += 1
            if key_offset >= key_length:
                key_offset = 0
            key_box[i] = key_box[c]
            key_box[c] = swap
            last_byte = c
        meta_length = f.read(4)
        meta_length = struct.unpack('<I', bytes(meta_length))[0]
        meta_data = f.read(meta_length)
        
        meta_data_array = bytearray(meta_data)
        for i in range(0, len(meta_data_array)):
            meta_data_array[i] ^= 0x63
        meta_data = bytes(meta_data_array)
        meta_data = base64.b64decode(meta_data[22:])
        cryptor = AES.new(meta_key, AES.MODE_ECB)
        meta_data = unpad(cryptor.decrypt(meta_data)).decode('utf-8')[6:]
        meta_data = json.loads(meta_data)
        crc32 = f.read(4)
        crc32 = struct.unpack('<I', bytes(crc32))[0]
        f.seek(5, 1)
        image_size = f.read(4)
        image_size = struct.unpack('<I', bytes(image_size))[0]
        image_data = f.read(image_size)
        file_name = os.path.basename(file_path).split(".ncm")[0] + '.' + meta_data['format']
        
        if output_dir is not None:
            output_path = os.path.join(output_dir, file_name)
        else:
            output_path = os.path.join(os.path.dirname(file_path), file_name)
        if not_overwrite and os.path.exists(output_path):
            print(f"Skipping {file_path}: output file (or a file with the same file name) exists")
            if del_ncm:
                os.remove(file_path)
            return
        with open(output_path, 'wb') as m:
            block_size = 1024 * 1024 * 10  # 10MB块大小
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                    
                # 根据设置选择解密方式
                if use_gpu and HAVE_PYCUDA:
                    try:
                        decrypted_chunk = gpu_decrypt(chunk, key_box)
                    except Exception as e:
                        print(f"GPU解密失败，回退到CPU: {e}")
                        decrypted_chunk = cpu_decrypt(chunk, key_box)
                else:
                    decrypted_chunk = cpu_decrypt(chunk, key_box)
                
                m.write(decrypted_chunk)
        
        if del_ncm:
            os.remove(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .ncm files to other formats')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', help='Input .ncm file or directory')
    group.add_argument('--file_list', help='File containing list of .ncm files')
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-d', '--del_ncm', help='Delete original .ncm files', action='store_true')
    parser.add_argument('--disable_gpu', help='Disable GPU acceleration', action='store_true')
    parser.add_argument('-n', '--not-overwrite', help='Skip if output file exists', action='store_true')
    args = parser.parse_args()

    # 确定是否使用GPU
    use_gpu = not args.disable_gpu and HAVE_PYCUDA
    if use_gpu:
        print("Using GPU acceleration")
    elif args.disable_gpu:
        print("GPU acceleration banned")
    elif not HAVE_PYCUDA:
        print("WARNING: Did not detect PyCUDA; will use CPU instead")

    if args.input:
        if os.path.isfile(args.input):
            files_to_process = [args.input]
        elif os.path.isdir(args.input):
            files_to_process = glob.glob(os.path.join(args.input, '*.ncm'))
        else:
            print(f"ERROR: {args.input} is not a file or directory")
            sys.exit(1)
    elif args.file_list:
        try:
            with open(args.file_list, 'r') as fl:
                files_to_process = [line.strip() for line in fl]
        except FileNotFoundError:
            print(f"ERROR: file list {args.file_list} cannot be found or accessed")
            sys.exit(1)
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in files_to_process:
        try:
            dump(file_path, output_dir, del_ncm=args.del_ncm, use_gpu=use_gpu, not_overwrite=args.not_overwrite)
            print(f"Successful: {file_path}")
        except Exception as e:
            print(f"ERROR while processing {file_path}: {e}")