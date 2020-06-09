#coding:utf-8
from func import *
from raw_graphs import *
import idc
import idaapi
import idautils
import os
import argparse


def parse_command():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument("--path", type=str, help="The directory where to store the generated .ida file")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	idaapi.autoWait()
	#args = parse_command()
	#path = args.path

	print "open a new binary"
	path = "D:\\Code\\python\\binary_search\\Genius\\Gencoding-master\\raw-feature-extractor\\extracted-acfg"
	
	analysis_flags = get_inf_attr(idc.INF_AF) # 得到analysis_flags  INF_AF
	analysis_flags &= ~AF_IMMOFF  # AF_IMMOFF ：将32位的指令操作转换为偏移量
	# turn off "automatically make offset" heuristic
	idc.set_inf_attr(INF_AF, analysis_flags) #设置analysis_flags
	cfgs = get_func_cfgs_c(FirstSeg())
	binary_name = GetInputFile() + '.cfg'
	fullpath = os.path.join(path, binary_name)
	pickle.dump(cfgs, open(fullpath,'w'))
	print binary_name

	idc.Exit(0)
