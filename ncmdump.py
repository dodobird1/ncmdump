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

def dump(file_path, output_dir=None, del_ncm=False, not_overwrite=False):
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
            while True:
                chunk = bytearray(f.read(0x8000))
                if not chunk:
                    break
                for i in range(1, len(chunk) + 1):
                    j = i & 0xff
                    chunk[i - 1] ^= key_box[(key_box[j] + key_box[(key_box[j] + j) & 0xff]) & 0xff]
                m.write(chunk)
        
        if del_ncm:
            os.remove(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .ncm files to other formats')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', help='Input .ncm file or directory')
    group.add_argument('--file_list', help='File containing list of .ncm files')
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-d', '--del_ncm', help='Delete original .ncm files', action='store_true')
    parser.add_argument('-n', '--not-overwrite', help='Skip if output file exists', action='store_true')
    args = parser.parse_args()

    if args.input:
        if os.path.isfile(args.input):
            files_to_process = [args.input]
        elif os.path.isdir(args.input):
            files_to_process = glob.glob(os.path.join(args.input, '*.ncm'))
        else:
            print(f"Error: {args.input} is neither a file nor a directory")
            sys.exit(1)
    elif args.file_list:
        try:
            with open(args.file_list, 'r') as fl:
                files_to_process = [line.strip() for line in fl]
        except FileNotFoundError:
            print(f"Error: File list {args.file_list} not found")
            sys.exit(1)
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in files_to_process:
        try:
            dump(file_path, output_dir, del_ncm=args.del_ncm, not_overwrite=args.not_overwrite)
            print(f"Successfully processed {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
