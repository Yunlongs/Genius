#coding:utf-8
#
# Reference Lister
#
# List all functions and all references to them in the current section.
#
# Implemented with the idautils module
#
from idautils import *
from idaapi import *
from idc import *
from idc_bc695 import *
import networkx as nx
import cfg_constructor as cfg
# import cPickle as pickle # in py3 there is no cPickle
import _pickle as pickle
import pdb
from raw_graphs import *
#from discovRe_feature.discovRe import *
from discovRe import *
#import wingdbstub
#wingdbstub.Ensure()
def gt_funcNames(ea):
	funcs = []
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(seg_start(ea)):
			funcname = get_unified_funcname(funcea)
			if funcname in plt_func:
				print(funcname)
				continue
			funcs.append(funcname)
	return funcs

def get_funcs(ea):
	funcs = {}
		# Get current ea
		# Loop from start to end in the current segment
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(seg_start(ea)):
		funcname = get_unified_funcname(funcea)
		if funcname in plt_func:
			continue
		func = get_func(funcea)
		blocks = FlowChart(func)
		funcs[funcname] = []
		for bl in blocks:
				start = bl.start_ea
				end = bl.end_ea
				funcs[funcname].append((start, end))
	return funcs

# used for the callgraph generation.
def get_func_namesWithoutE(ea):
	funcs = {}
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(seg_start(ea)):
			funcname = get_unified_funcname(funcea)
			if 'close' in funcname:
				print(funcea)
			if funcname in plt_func:
				print(funcname)
				continue
			funcs[funcname] = funcea
	return funcs

# used for the callgraph generation.
def get_func_names(ea):
	funcs = {}
	for funcea in Functions(seg_start(ea)):
			funcname = get_unified_funcname(funcea)
			funcs[funcname] = funcea
	return funcs

def get_func_bases(ea):
		funcs = {}
		plt_func, plt_data = processpltSegs()
		for funcea in Functions(seg_start(ea)):
				funcname = get_unified_funcname(funcea)
				if funcname in plt_func:
					continue
				funcs[funcea] = funcname
		return funcs

def get_func_range(ea):
		funcs = {}
		for funcea in Functions(seg_start(ea)):
				funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		funcs[funcname] = (func.start_ea, func.end_ea)
		return funcs

def get_unified_funcname(ea): # 得到统一形式的functionName
	funcname = GetFunctionName(ea)
	if len(funcname) > 0:
		if '.' == funcname[0]:
			funcname = funcname[1:]
	return funcname

def get_func_sequences(ea):
	funcs_bodylist = {}
	funcs = get_funcs(ea)
	for funcname in funcs:
		if funcname not in funcs_bodylist:
			funcs_bodylist[funcname] = []
		for start, end in funcs[funcname]:
			inst_addr = start
			while inst_addr <= end:
				opcode = GetMnem(inst_addr)
				funcs_bodylist[funcname].append(opcode)
				inst_addr = NextHead(inst_addr)
	return funcs_bodylist

def get_func_cfgs_c(ea):
	'''
	ea:binary的起始地址
	return: 每个函数的原生属性控制流图（未向量化）的列表
	'''
	# binary_name = GetInputFile()
	binary_name = idc.get_root_filename()
	print("+_+_+_+_+_+_+_+_+_")
	print(binary_name, type(binary_name))
	print("+_+_+_+_+_+_+_+_+_")
	raw_cfgs = raw_graphs(binary_name)
	externs_eas, ea_externs = processpltSegs()
	i = 0
	segm = get_segm_by_name(".text")
	for funcea in Functions(segm.start_ea,segm.end_ea):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea) # 得到func这个类对象
		#print i
		i += 1
		icfg = cfg.getCfg(func, externs_eas, ea_externs)  # 为每个函数构建Genius 的ACFG
		func_f = get_discoverRe_feature(func, icfg)  # 生成DiscoverRe中的函数特征
		raw_g = raw_graph(funcname, icfg, func_f)
		raw_cfgs.append(raw_g)

	return raw_cfgs

def get_func_cfgs_ctest(ea):
	# binary_name = idc.GetInputFile()
	binary_name = idc.get_input_file_path
	raw_cfgs = raw_graphs(binary_name)
	externs_eas, ea_externs = processpltSegs()
	i = 0
	diffs = {}
	for funcea in Functions(seg_start(ea)):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		print(i)
		i += 1
		icfg, old_cfg = cfg.getCfg(func, externs_eas, ea_externs)
		diffs[funcname] = (icfg, old_cfg)
		#raw_g = raw_graph(funcname, icfg)
		#raw_cfgs.append(raw_g)
			
	return diffs

def get_func_cfgs(ea):
	func_cfglist = {}
	i = 0
	for funcea in Functions(seg_start(ea)):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		print(i)
		i += 1
		try:
			icfg = cfg.getCfg(func)
			func_cfglist[funcname] = icfg
		except:
			pass
			
	return func_cfglist

def get_func_cfg_sequences(func_cfglist):
	func_cfg_seqlist = {}
	for funcname in func_cfglist:
		func_cfg_seqlist[funcname] = {}
		cfg = func_cfglist[funcname][0]
		for start, end in cfg:
			codesq = get_sequences(start, end)
			func_cfg_seqlist[funcname][(start,end)] = codesq

	return func_cfg_seqlist


def get_sequences(start, end):
	seq = []
	inst_addr = start
	while inst_addr <= end:
		opcode = GetMnem(inst_addr)
		seq.append(opcode)
		inst_addr = NextHead(inst_addr)
	return seq

def get_stack_arg(func_addr):
	print(func_addr)
	args = []
	stack = GetFrame(func_addr)
	if not stack:
			return []
	firstM = GetFirstMember(stack)
	lastM = GetLastMember(stack)
	i = firstM
	while i <=lastM:
		mName = GetMemberName(stack,i)
		mSize = GetMemberSize(stack,i)
		if mSize:
				i = i + mSize
		else:
				i = i+4
		if mName not in args and mName and ' s' not in mName and ' r' not in mName:
			args.append(mName)
	return args

		#pickle.dump(funcs, open('C:/Documents and Settings/Administrator/Desktop/funcs','w'))

def processExternalSegs():
	funcdata = {}
	datafunc = {}
	for n in range(get_segm_qty()):
		seg =getnseg(n)
		ea = seg.start_ea
		segtype = idc.GetSegmentAttr(ea, idc.SEGATTR_TYPE)
		if segtype in [idc.SEG_XTRN]:
			start = idc.seg_start(ea)
			end = idc.SegEnd(ea)
			cur = start
			while cur <= end:
				name = get_unified_funcname(cur)
				funcdata[name] = hex(cur)
				cur = NextHead(cur)
	return funcdata

def processpltSegs(): # 得到所有函数对应的起始指令地址，和得到所有函数起始地址对应的函数名
	funcdata = {}
	datafunc = {}
	for n in range(get_segm_qty()): #Segment总数
		seg = getnseg(n)
		# ea = seg.start_ea
		ea = seg.start_ea
		segname = SegName(ea)
		if segname in ['.plt', 'extern', '.MIPS.stubs']:
			start = seg.start_ea
			end = seg.end_ea
			cur = start
			while cur < end:
				name = get_unified_funcname(cur)
				funcdata[name] = hex(cur)
				datafunc[cur]= name
				cur = NextHead(cur)
	return funcdata, datafunc

		
def processDataSegs():
	funcdata = {}
	datafunc = {}
	for n in range(get_segm_qty()):
		seg = getnseg(n)
		ea = seg.start_ea
		segtype = idc.GetSegmentAttr(ea, idc.SEGATTR_TYPE)
		if segtype in [idc.SEG_DATA, idc.SEG_BSS]:
			start = idc.seg_start(ea)
			end = idc.SegEnd(ea)
			cur = start
			while cur <= end:
				refs = [v for v in DataRefsTo(cur)]
				for fea in refs:
					name = get_unified_funcname(fea)
					if len(name)== 0:
						continue
					if name not in funcdata:
						funcdata[name] = [cur]
					else:
						funcdata[name].append(cur)
					if cur not in datafunc:
						datafunc[cur] = [name]
					else:
						datafunc[cur].append(name)
				cur = NextHead(cur)
	return funcdata, datafunc

def obtainDataRefs(callgraph):
	datarefs = {}
	funcdata, datafunc = processDataSegs()
	for node in callgraph:
		if node in funcdata:
			datas = funcdata[node]
			for dd in datas:
				refs = datafunc[dd]
				refs = list(set(refs))
				if node in datarefs:
					print(refs)
					datarefs[node] += refs
					datarefs[node] = list(set(datarefs[node]))
				else:
					datarefs[node] = refs
	return datarefs

