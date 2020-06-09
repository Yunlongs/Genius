import numpy as np
import networkx as nx
import pickle
from train import *

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

with open("output/centroid_cfgs","rb") as f:
    centroid_cfgs = pickle.load(f)
with open("output/query_cfgs","rb") as f:
    query_cfgs = pickle.load(f)

query_cfg = query_cfgs["openssl-101a_x86_gcc_O2_openssl-level_find_node"]
x,y = 1,len(centroid_cfgs)
matrix = np.zeros((2,y))
for i,query in enumerate([query_cfg]):
    for j,c in enumerate(centroid_cfgs.values()):
        weight = generate_weight(query.g,c.g)
        similarity = bipartite_match(weight).compute_graph_similarity()
        matrix[i,j]=similarity

with open("../raw-feature-extractor/extracted-acfg/openssl-101a_arm_clang_O0_openssl.cfg","r") as f:
    cfg = pickle.load(StrToBytes(f)).raw_graph_list
    for g in cfg:
        if g.funcname =="level_find_node":
            target = g
            break

for i,query in enumerate([target]):
    for j,c in enumerate(centroid_cfgs.values()):
        weight = generate_weight(query.g,c.g)
        similarity = bipartite_match(weight).compute_graph_similarity()
        matrix[1,j]=similarity

with open("../raw-feature-extractor/extracted-acfg/openssl-101f_mips_gcc_O1_openssl.cfg","r") as f:
    cfg = pickle.load(StrToBytes(f)).raw_graph_list
    for g in cfg:
        if g.funcname =="aep_init":
            target = g
            break
for i,query in enumerate([target]):
    for j,c in enumerate(centroid_cfgs.values()):
        weight = generate_weight(query.g,c.g)
        similarity = bipartite_match(weight).compute_graph_similarity()
        matrix[1,j]=similarity
print(123)