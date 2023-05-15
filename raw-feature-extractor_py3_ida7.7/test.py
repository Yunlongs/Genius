import pickle
from raw_graphs import *

with open(".\\extracted-acfg\\nbsmtp.cfg", 'rb') as f:
    tmp = pickle.load(f, encoding="bytes")
    print(tmp)
    print(tmp.binary_name)
    print(tmp.raw_graph_list)
    print(tmp.raw_graph_list[0])
    print("==x=x=x=x=xxx")
    print(tmp.raw_graph_list[0].funcname, '\n')
    print(tmp.raw_graph_list[0].g, type(tmp.raw_graph_list[0].g), '\n')
    print(tmp.raw_graph_list[0].discovre_features)


'''
    self.funcname = funcname
    self.old_g = g
    self.g = nx.DiGraph()
    self.discovre_features = func_f
    self.attributing()
'''