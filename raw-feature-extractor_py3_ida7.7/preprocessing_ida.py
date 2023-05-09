#coding:utf-8
from func import *
from raw_graphs import *
import idc
import idc_bc695
import idaapi
import idautils
import os
import argparse

print("test if run")


def parse_command():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument("--path", type=str, help="The directory where to store the generated .ida file")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# idaapi.autoWait()
	idaapi.auto_wait()
	#args = parse_command()
	#path = args.path

	print("open a new binary")
	# path = "D:\\Code\\python\\binary_search\\Genius\\Gencoding-master\\raw-feature-extractor\\extracted-acfg"
	path = "D:\\Desktop\\Genius-master\\raw-feature-extractor_trypy3\\extracted-acfg"
	analysis_flags = get_inf_attr(idc.INF_AF) # 得到analysis_flags  INF_AF
	analysis_flags &= ~AF_IMMOFF  # AF_IMMOFF ：将32位的指令操作转换为偏移量
	# turn off "automatically make offset" heuristic
	idc.set_inf_attr(INF_AF, analysis_flags) #设置analysis_flags

	# cfgs = get_func_cfgs_c(FirstSeg())
	# get_first_seg()
	cfgs = get_func_cfgs_c(get_first_seg())

	binary_name = GetInputFile() + '.cfg'
	fullpath = os.path.join(path, binary_name)
	print(cfgs)
	print("===================--====================")
	pickle.dump(cfgs, open(fullpath,'wb'))
	print(binary_name)

	ida_pro.qexit(0)
