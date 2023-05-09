#coding:utf-8

import copy
import networkx as nx
from idautils import *
import idaapi
from idc import *
from idc_bc695 import *
from graph_analysis_ida import *


def getCfg(func, externs_eas, ea_externs):
	func_start = func.start_ea
	func_end = func.end_ea
	cfg = nx.DiGraph()
	control_blocks, main_blocks = obtain_block_sequence(func)
	i = 0
	visited = {}
	start_node = None
	for bl in control_blocks:
		start = control_blocks[bl][0]
		end = control_blocks[bl][1]
		src_node = (start, end)
		if src_node not in visited:
			src_id = len(cfg)
			visited[src_node] = src_id
			cfg.add_node(src_id)
			cfg.nodes[src_id]['label'] = src_node

		else:
			src_id = visited[src_node]

		#if end in seq_blocks and GetMnem(prev_head(end)) != 'jmp':
		if start == func_start:
			cfg.nodes[src_id]['c'] = "start"
			start_node = src_node
		if end == func_end:
			cfg.nodes[src_id]['c'] = "end"
		#print control_ea, 1
		refs = CodeRefsTo(start, 0)
		for ref in refs:
			if ref in control_blocks:
				dst_node = control_blocks[ref]
				if dst_node not in visited:
					visited[dst_node] = len(cfg)
				dst_id = visited[dst_node]
				cfg.add_edge(dst_id, src_id)
				cfg.nodes[dst_id]['label'] = dst_node
		#print control_ea, 1
		refs = CodeRefsTo(start, 1)  #除了跳转Flow，将正常FLOW也算上
		for ref in refs:
			if ref in control_blocks:
				dst_node = control_blocks[ref]
				if dst_node not in visited:
					visited[dst_node] = len(cfg)
				dst_id = visited[dst_node]
				cfg.add_edge(dst_id, src_id)
				cfg.nodes[dst_id]['label'] = dst_node
	#print "attributing"
	cfg = attributingRe(cfg, externs_eas, ea_externs)
	# removing deadnodes
	#old_cfg = copy.deepcopy(cfg)
	#transform(cfg)
	return cfg

def transform(cfg):
	merging(cfg)
	filtering(cfg)

def merging(cfg):
	bb_ids = cfg.nodes()
	for bb_id in bb_ids:
		try:
			bb = cfg.nodes[bb_id]['label']
			bb_start = bb[0]
			bb_end = bb[1]
			succs = cfg.successors(bb_id)
			#preds = cfg.predecessors(bb_id)
			if len(succs) == 1:
				preds = cfg.predecessors(succs[0])
				if len(preds) == 1:
					domerge(cfg, bb_id, succs[0])
		except:
			pass

def domerge(cfg, bb_id, suc_node):
	suc_nodes = cfg.successors(suc_node)
	for node in suc_nodes:
		cfg.add_edge(bb_id, node)
	cfg.remove_node(suc_node)


def filtering(cfg):
	rm_sets = []
	for bb_id in cfg:
		bb = cfg.nodes[bb_id]['label']
		bb_start = bb[0]
		bb_end = bb[1]
		re = remove(bb_start, bb_end)
		print(bb_id, re, bb_start, bb_end)
		if re:
			print(re, bb_id)
			rm_sets.append(bb_id)
	print(rm_sets)
	for bb_id in rm_sets:
		cfg.remove_node(bb_id)

def remove(bb_start, bb_end):
	seqs = getSequences(bb_start, bb_end)
	if matchseq(seqs):
		return True
	return False

def matchseq(seqs):
	mips = set(['lw', "jr", "addiu"])
	x86 = set(['add', 'pop', 'retn'])
	b_mips = set(['b', ('move','$v0')])
	b_x86 = set(['b', ('mov','$eax')])
	re_mips = set([('move','$v0')])
	re_x86 = set([('mov','$eax')])
	diff_mips = set(seqs).difference(set(mips))
	if len(diff_mips) == 0:
		return True
	diff_x86 = set(seqs).difference(set(x86))
	if len(diff_x86) == 0:
		return True
	if set(seqs) == b_mips:
		return True
	if set(seqs) == b_x86:
		return True
	if set(seqs) == re_mips:
		return True
	if set(seqs) == re_x86:
		return True
	return False

def attributingRe(cfg, externs_eas, ea_externs):  #为每个基本块生成自定义的属性
	for node_id in cfg:
		bl = cfg.nodes[node_id]['label']
		numIns = calInsts(bl)  # No. of Instruction
		cfg.nodes[node_id]['numIns'] = numIns
		numCalls = calCalls(bl) # No. of Calls
		cfg.nodes[node_id]['numCalls'] = numCalls
		numLIs = calLogicInstructions(bl)  # 这个不再Genius的范围内
		cfg.nodes[node_id]['numLIs'] = numLIs
		numAs = calArithmeticIns(bl) # No. of Arithmetic Instructions
		cfg.nodes[node_id]['numAs'] = numAs
		strings, consts = getBBconsts(bl)  # String and numeric constants
		cfg.nodes[node_id]['numNc'] = len(strings) + len(consts)
		cfg.nodes[node_id]['consts'] = consts
		cfg.nodes[node_id]['strings'] = strings
		externs = retrieveExterns(bl, ea_externs)
		cfg.nodes[node_id]['externs'] = externs
		numTIs = calTransferIns(bl) # No. of Transfer Instruction
		cfg.nodes[node_id]['numTIs'] = numTIs
	return cfg



def attributing(cfg):
	ga = graph_analysis()
	ga.gwithoffspring(cfg)
	print("finishing offspring")
	for node in cfg:
		stmt_num = getStmtNum(node)
		binary_value = getBinaryValue(node)
		cfg.nodes[node]['stmt_num'] = stmt_num
		cfg.nodes[node]['binary_value'] = binary_value
	ga.domChecking(cfg)
	print("finishing domChecking")
	ga.loopChecking(cfg)
	print("finishing loopChecking")


def getStmtNum(node):
	start = node[0]
	end = node[1]
	stmt_num = 0
	inst_addr = start
	while inst_addr < end:
		inst_addr = NextHead(inst_addr)
		stmt_num += 1
	return stmt_num

def getBinaryValue(node):
	start = node[0]
	inst_addr = NextHead(start)
	value = 0
	addr = 0
	for x in range((inst_addr - start)-1):
		addr = start + x
		y = GetOriginalByte(addr)
		print(value, addr, y)
		value = value | y
		value = value << 8
		print(value)

	addr = inst_addr - 1
	y = GetOriginalByte(addr)
	print(value, addr, y)
	value = value | y
	print(node)
	print(bin(value))
	return value


def cfg_construct(func):
	func_start = func.start_ea
	func_end = func.end_ea
	cfg = nx.DiGraph()
	seq_blocks, main_blocks = obtain_block_sequence(func)
	i = 0
	visited = {}
	for bl in seq_blocks:
		start = seq_blocks[bl][0]
		end = seq_blocks[bl][1]
		src_node = (start, end)
		if end in seq_blocks and GetMnem(prev_head(end)) != 'jmp':
						next_start = seq_blocks[end][0]
						next_end = seq_blocks[end][1]
						next_node = (next_start, next_end)
						cfg.add_edge(src_node, next_node)
		if start == func_start:
			cfg.add_node(src_node, c='start')
			start_node = src_node
		if end == func_end:
			cfg.add_node(src_node, c='end')
		refs = CodeRefsFrom(prev_head(end), 0)
		
		for ref in refs:
						#print ref
						if ref in seq_blocks:
								dst_node = (seq_blocks[ref][0], seq_blocks[ref][1])
								cfg.add_edge(src_node, dst_node)
	return cfg, start_node


def obtain_allpaths( cfg, node, path, allpaths):
	path.append(node)
	if 'c' in cfg.nodes[node] and cfg.nodes[node]['c'] == 'end':
		allpaths.append(path)
		return
	else:
		for suc in cfg.successors(node):
						if suc not in path:
								path_copy = copy.copy(path)
								obtain_allpaths(cfg, suc, path_copy, allpaths)


def obtain_block_sequence(func):
	control_blocks = {}
	main_blocks = {}
	blocks = [(v.start_ea, v.end_ea) for v in idaapi.FlowChart(func)]  # 返回该函数所有的basicblock
	for bl in blocks:
		base = bl[0]
		end = prev_head(bl[1])
		control_ea = checkCB(bl)
		control_blocks[control_ea] = bl
		control_blocks[end] = bl
		if func.start_ea <= base <= func.end_ea:
						main_blocks[base] = bl
		x = sorted(main_blocks)
	return control_blocks, x

def checkCB(bl): # 检查基本块的正确性
	start = bl[0]
	end = bl[1]
	ea = start
	while ea < end:
		if checkCondition(ea):
			return ea
		ea = NextHead(ea)

	return prev_head(end)

def checkCondition(ea): # check是否是跳转指令
	mips_branch = {"beqz":1, "beq":1, "bne":1, "bgez":1, "b":1, "bnez":1, "bgtz":1, "bltz":1, "blez":1, "bgt":1, "bge":1, "blt":1, "ble":1, "bgtu":1, "bgeu":1, "bltu":1, "bleu":1}
	x86_branch = {"jz":1, "jnb":1, "jne":1, "je":1, "jg":1, "jle":1, "jl":1, "jge":1, "ja":1, "jae":1, "jb":1, "jbe":1, "jo":1, "jno":1, "js":1, "jns":1}
	arm_branch = {"B":1, "BAL":1, "BNE":1, "BEQ":1, "BPL":1, "BMI":1, "BCC":1, "BLO":1, "BCS":1, "BHS":1, "BVC":1, "BVS":1, "BGT":1, "BGE":1, "BLT":1, "BLE":1, "BHI":1 ,"BLS":1 }
	conds = {}
	conds.update(mips_branch)
	conds.update(x86_branch)
	opcode = GetMnem(ea)
	if opcode in conds:
		return True
	return False