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
import platform
import time
from enum import Enum
import numpy as np

class AccelerationMode(Enum):
    AUTO = 'auto'
    CUDA = 'cuda'
    METAL = 'metal'
    CPU = 'cpu'

# 尝试导入CUDA相关库
HAVE_CUDA = False
if platform.system() == 'Windows' or platform.system() == 'Linux':
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        HAVE_CUDA = True
    except ImportError:
        pass

# 尝试导入Metal相关库
HAVE_METAL = False
if platform.system() == 'Darwin':
    try:
        import metal
        from metal import MTLCreateSystemDefaultDevice, MTLResourceStorageModeShared
        HAVE_METAL = True
    except ImportError:
        pass

def cuda_decrypt(data, key_box):
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
    kernel = mod.get_function("decrypt_kernel")
    kernel(data_gpu, key_box_gpu, np.int32(n),
           block=(threads_per_block, 1, 1),
           grid=(blocks_per_grid, 1))
    
    # 将结果复制回主机
    result = bytearray(n)
    cuda.memcpy_dtoh(result, data_gpu)
    return bytes(result)

def metal_decrypt(data, key_box):
    """使用Apple Metal GPU解密数据"""
    n = len(data)
    # 创建Metal设备
    device = MTLCreateSystemDefaultDevice()
    command_queue = device.newCommandQueue()
    
    # 创建Metal缓冲区
    data_buffer = device.newBufferWithBytes_length_options_(data, n, MTLResourceStorageModeShared)
    key_box_buffer = device.newBufferWithBytes_length_options_(key_box, 256, MTLResourceStorageModeShared)
    
    # 创建计算管道
    source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void decrypt_kernel(device uchar *data [[buffer(0)]],
                              constant uchar *key_box [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
        if (id < %d) {
            uchar j = id & 0xff;
            uchar a = key_box[j];
            uchar b = key_box[(a + j) & 0xff];
            data[id] ^= key_box[(a + b) & 0xff];
        }
    }
    """ % n
    
    # 编译Metal函数
    library = device.newLibraryWithSource_options_error_(source, None, None)
    function = library.newFunctionWithName_("decrypt_kernel")
    compute_pipeline = device.newComputePipelineStateWithFunction_error_(function, None)
    
    # 创建命令缓冲区
    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()
    
    # 设置参数
    compute_encoder.setComputePipelineState_(compute_pipeline)
    compute_encoder.setBuffer_atIndex_(data_buffer, 0, 0)
    compute_encoder.setBuffer_atIndex_(key_box_buffer, 0, 1)
    
    # 配置线程
    grid_size = metal.MTLSize(n, 1, 1)
    threadgroup_size = metal.MTLSize(min(compute_pipeline.maxTotalThreadsPerThreadgroup(), 512), 1, 1)
    compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    compute_encoder.endEncoding()
    
    # 执行命令
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    # 获取结果
    result = bytes(data_buffer.contents()[:n])
    return result

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

def dump(file_path, output_dir=None, del_ncm=False, accel_mode=AccelerationMode.AUTO):
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
        
        # 确定加速方式
        actual_accel = accel_mode
        if accel_mode == AccelerationMode.AUTO:
            if HAVE_CUDA:
                actual_accel = AccelerationMode.CUDA
            elif HAVE_METAL:
                actual_accel = AccelerationMode.METAL
            else:
                actual_accel = AccelerationMode.CPU
        
        # 打印加速方式信息
        accel_info = {
            AccelerationMode.CUDA: "NVIDIA CUDA GPU加速",
            AccelerationMode.METAL: "Apple Metal GPU加速",
            AccelerationMode.CPU: "CPU解密"
        }
        print(f"处理 {os.path.basename(file_path)}: 使用{accel_info[actual_accel]}")
        
        start_time = time.time()
        with open(output_path, 'wb') as m:
            block_size = 1024 * 1024 * 10  # 10MB块大小
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                    
                # 根据加速方式选择解密方法
                try:
                    if actual_accel == AccelerationMode.CUDA:
                        decrypted_chunk = cuda_decrypt(chunk, key_box)
                    elif actual_accel == AccelerationMode.METAL:
                        decrypted_chunk = metal_decrypt(chunk, key_box)
                    else:
                        decrypted_chunk = cpu_decrypt(chunk, key_box)
                except Exception as e:
                    print(f"硬件加速失败，回退到CPU: {e}")
                    decrypted_chunk = cpu_decrypt(chunk, key_box)
                
                m.write(decrypted_chunk)
        
        process_time = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"处理完成: {file_size:.2f}MB, 耗时: {process_time:.2f}秒, 速度: {file_size/process_time:.2f} MB/s")
        
        if del_ncm:
            os.remove(file_path)

# 全局CUDA内核（如果可用）
mod = None
if HAVE_CUDA:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .ncm files to other formats')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', help='Input .ncm file or directory')
    group.add_argument('--file_list', help='File containing list of .ncm files')
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-d', '--del_ncm', help='Delete original .ncm files', action='store_true')
    parser.add_argument('--accel', help='Acceleration mode (auto, cuda, metal, cpu)', 
                        choices=['auto', 'cuda', 'metal', 'cpu'], default='auto')
    args = parser.parse_args()

    # 打印可用加速选项
    print("可用硬件加速选项:")
    print(f"- NVIDIA CUDA: {'可用' if HAVE_CUDA else '不可用'}")
    print(f"- Apple Metal: {'可用' if HAVE_METAL else '不可用'}")
    print(f"选择的加速模式: {args.accel}")

    # 映射加速模式
    accel_mode = AccelerationMode(args.accel)

    if args.input:
        if os.path.isfile(args.input):
            files_to_process = [args.input]
        elif os.path.isdir(args.input):
            files_to_process = glob.glob(os.path.join(args.input, '*.ncm'))
        else:
            print(f"错误: {args.input} 不是文件或目录")
            sys.exit(1)
    elif args.file_list:
        try:
            with open(args.file_list, 'r') as fl:
                files_to_process = [line.strip() for line in fl]
        except FileNotFoundError:
            print(f"错误: 文件列表 {args.file_list} 未找到")
            sys.exit(1)
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 处理文件
    total_start = time.time()
    for file_path in files_to_process:
        try:
            dump(file_path, output_dir, del_ncm=args.del_ncm, accel_mode=accel_mode)
        except Exception as e:
            print(f"处理失败 {file_path}: {e}")
    
    total_time = time.time() - total_start
    print(f"所有文件处理完成! 总耗时: {total_time:.2f}秒")