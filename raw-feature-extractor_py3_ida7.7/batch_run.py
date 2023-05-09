#coding:utf-8
import os
import subprocess
import glob
import time

IDA_PATH = "D:\\IStoolkit\\IDAPro7.7\\ida64.exe"
PLUGIN_PATH = "D:\\Desktop\\Genius-master\\raw-feature-extractor_trypy3\\preprocessing_ida.py"  #必须绝对路径

# 获取所有需要分析的二进制文件路径
ELF_PATH = glob.glob("D:\\Desktop\\Genius-master\\raw-feature-extractor_trypy3\\test\\*")
print(ELF_PATH)
DST_path = "D:\\Desktop\\Genius-master\\raw-feature-extractor_trypy3\\extracted-acfg"

core_num = 6  ##并行的个数，或者电脑有几个核心
start = 0
for elf in ELF_PATH:
    if elf.endswith(".i64"):
        continue
    # cmd = IDA_PATH + " -c -A -S"+PLUGIN_PATH+" "+elf
    cmd = IDA_PATH + " -c -S" + PLUGIN_PATH + " " + elf
    print(cmd)
    if start >= core_num:
        finish = len(os.listdir(DST_path))
        while start-core_num >= finish:
            time.sleep(1)
            finish = len(os.listdir(DST_path))
            print(start, finish)

    subprocess.Popen(cmd)
    start += 1