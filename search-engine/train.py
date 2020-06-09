# coding:utf-8
import numpy as np
import pickle
import time
import glob
import networkx as nx
from networkx.algorithms import bipartite
import os
import concurrent.futures
import gc

## The Km bipartite match algoritm achieved by python.but profermance too slow,not to recommend
class KM:
    def __init__(self, weight):
        self.weight = -weight
        self.nx, self.ny = weight.shape
        self.lx = np.zeros((self.nx,))
        self.ly = np.zeros((self.ny,))
        self.inf = 0x3f3f3f3f
        self.slack = np.zeros((self.ny,))
        self.match = -np.ones((self.ny,), dtype=int)

    def init(self):
        for x in range(self.nx):
            self.lx[x] = np.max(self.weight[x])

    def findpath(self, x, visx, visy):
        visx[x] = 1
        for y in range(self.ny):
            if not visy[y] and self.lx[x] + self.ly[y] == self.weight[x][y]:
                visy[y] = 1
                if self.match[y] == -1 or self.findpath(self.match[y], visx, visy):
                    self.match[y] = x
                    return True
        return False

    def Max_match(self):
        self.init()
        for x in range(self.nx):
            while True:
                visx = np.zeros((self.nx,))
                visy = np.zeros((self.ny,))
                if self.findpath(x, visx, visy):
                    break
                else:
                    delta = 0
                    for i in range(self.nx):
                        if visx[i]:
                            for j in range(self.ny):
                                if visy[j]:
                                    delta = min(delta, self.lx[i] + self.lx[j] - self.weight[i, j])
                    for i in range(self.nx):
                        if visx[i]:
                            self.lx[i] -= delta
                    for j in range(self.ny):
                        if visy[j]:
                            self.ly[j] += delta

    def findpath2(self, x, visx, visy):
        visx[x] = 1
        for y in range(self.ny):
            if visy[y]:
                continue
            tempDelta = self.lx[x] + self.ly[y] - self.weight[x, y]
            if tempDelta == 0:
                visy[y] = 1
                if self.match[y] == -1 or self.findpath2(self.match[y], visx, visy):
                    self.match[y] = x
                    return True
            elif self.slack[y] > tempDelta:
                self.slack[y] = tempDelta
        return False

    def improved_KM(self):
        for x in range(self.nx):
            for j in range(self.ny):
                self.slack[j] = self.inf
            while True:
                visx = np.zeros((self.nx,))
                visy = np.zeros((self.ny,))
                if self.findpath2(x, visx, visy):
                    break
                else:
                    delta = self.inf
                    for j in range(self.ny):
                        if not visy[j] and delta > self.slack[j]:
                            delta = self.slack[j]
                    for i in range(self.nx):
                        if visx[i]:
                            self.lx[i] -= delta
                    for j in range(self.ny):
                        if visy[j]:
                            self.ly[j] += delta
                        else:
                            self.slack[j] -= delta

    def compute_graph_similarity(self):
        self.improved_KM()
        score = 0
        self.weight = -self.weight
        for y, x in enumerate(self.match):
            if x != -1:
                score += self.weight[x, y]
        similarity = 1 - score / max(self.nx, self.ny)
        return similarity

## bipartite mathch algorithm achieved by libary of networkx, actually use scipy,fast.
class bipartite_match(object):
    def __init__(self, weight):
        self.weiht = weight
        self.nx, self.ny = weight.shape
        self.G = nx.Graph()
        for x in range(self.nx):
            self.G.add_node(x, bipartite=0)
        for y in range(self.ny):
            self.G.add_node(10000 + y, bipartite=1)

        for x in range(self.nx):
            for y in range(self.ny):
                self.G.add_edge(x, 10000 + y, weight=weight[x, y])

    def compute_graph_similarity(self):
        match = bipartite.minimum_weight_full_matching(self.G)
        score = 0
        for x in range(self.nx):
            score += self.weiht[x, match[x] - 10000]
        ## there are three graph similarity compute method,please choose one

        ### the fisrt :author's paper's formula
        # similarity = 1 - score/max(self.nx,self.ny)

        ### the second: fix the max to min

        similarity = 1 - score / min(self.nx, self.ny)

        ### the third: improved second'formula by normalize node's effective
        # similarity = 1 - score / min(self.nx, self.ny)
        # similarity = similarity * self.nx / self.ny
        del self.G
        del self.weiht
        return similarity


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# weight = np.array(([[1,1,1,1,2,1],[2,3,5,2,1,2],[1,3,2,9,2,5],[1,1,1,1,1,6],[5,2,2,3,4,2]]))
# print (weight)

def block_distance(feature1, feature2):
    top = bottom = 0
    alpha = [10.82, 14.47, 198.67, 55.65, 66.22, 41.37, 5, 6.54]
    for i in range(2):
        top += (len(set.union(set(feature1[i]), set(feature2[i]))) - len(
            set.intersection(set(feature1[i]), set(feature2[i])))) * alpha[i]
        bottom += len(set.union(set(feature1[i]), set(feature2[i]))) * alpha[i]
    for i in range(2, 8):
        top += np.abs(feature1[i] - feature2[i]) * alpha[i]
        bottom += max(feature1[i], feature2[i]) * alpha[i]
    if bottom == 0:
        return 0
    return top / bottom


def generate_weight(func1, func2):
    nodes1 = len(func1.nodes)
    nodes2 = len(func2.nodes)
    nx = nodes1
    ny = nodes2
    if nodes2 < nodes1:
        nx = nodes2
        ny = nodes1
        func = func1
        func1 = func2
        func2 = func
    weight = np.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            weight[x, y] = block_distance(func1.nodes[x]['v'], func2.nodes[y]['v'])
    return weight


def load_two_functions():
    picklefile = open("../raw-feature-extractor/extracted-acfg/x86-gcc-openssl-O0.cfg", "r")
    cfg1 = pickle.load(StrToBytes(picklefile))
    picklefile = open("../raw-feature-extractor/extracted-acfg/x86-clang-openssl-O0.cfg", "r")
    cfg2 = pickle.load(StrToBytes(picklefile))
    cfg1 = cfg1.raw_graph_list
    cfg2 = cfg2.raw_graph_list
    func_gcc = cfg1[5]  # 拿对应的main函数做实验
    func_clang = cfg2[5]
    print(func_gcc.funcname, func_clang.funcname)
    return func_gcc.g, func_clang.g


def compute_one_to_all(arg_list):
    i, func_num, Acfg_list = arg_list
    vector = np.zeros(func_num)
    for j in range(i,func_num):
        func1, func2 = Acfg_list[i], Acfg_list[j]
        weight = generate_weight(func1, func2)
        similarity = bipartite_match(weight).compute_graph_similarity()
        vector[j] = similarity
        del weight
    gc.collect()
    return vector



def random_sample(funcname_list):
    '''
    random select 1w function as baseline dataset
    '''
    func_len = len(funcname_list)
    print("The number of selected function:%s" % func_len)

    assert func_len >= 10000, "The number of functions smaller than 10000"
    sampled_func = np.random.choice(funcname_list,10000,replace=False)

    ## record sampled labels and  the func who are single
    label ={}
    for i,func in enumerate(sampled_func):
        funcname = func.split('-')[-1]
        if label.get(funcname)==None:
            label[funcname] = 0
        label[funcname] +=1

    single_funcname=[]
    final_funcname=[]
    for i,func in enumerate(sampled_func):
        funcname = func.split('-')[-1]
        if label[funcname]==1:
            single_funcname.append(func)
        else:
            final_funcname.append(func)

    print("number of single func:%s"%len(single_funcname))
    ## find pair for the single func
    j = 0
    for i,func in enumerate(single_funcname):
        funcname = func.split('-')[-1]
        flag = 0
        for func2 in funcname_list:
            funcname2 = func2.split('-')[-1]
            if funcname ==funcname2 and func != func2:
                final_funcname.append(func2)
                final_funcname.append(func)
                flag =1
                break
        if flag==0:
            j +=1
            continue
        label[funcname]+=1
        if i > len(single_funcname)//2+j:
            break
    print("finally sampled number of function :%s"% len(final_funcname))

    ## then,random select 1k func for quering and others for trainning
    query = np.random.choice(final_funcname,1000,replace=False)
    for q in query:
        final_funcname.remove(q)
    print("Query set size:%s \n train set size:%s"%(len(query),len(final_funcname)))

    labels =[]
    for q in query:
        funcname = q.split('-')[-1]
        labels.append(label[funcname]-1)
    return query,final_funcname,labels


Acfg_path = "D:\\Code\\python\\binary_search\\Genius\\Gencoding-master\\raw-feature-extractor\\extracted-acfg/*"
max_limit = 100
min_limit = 3
labels_path = "output/label.npy"
query_path = "output/query.npy"
train_func_path = "output/train_func.txt"
large_func_path = "output/large_func.txt"
small_func_path = "output/small_func.txt"
kernel_matrix_path = "output/kernel_matrix.npy"


## for AI platfrom train
Acfg_path = "/data/Yunlong/extracted-acfg/extracted-acfg/*"
max_limit = 100
min_limit = 3
labels_path = "/output/label.npy"
query_path = "/output/query.npy"
train_func_path = "/output/train_func.txt"
large_func_path = "/output/large_func.txt"
small_func_path = "/output/small_func.txt"
kernel_matrix_path = "/output/kernel_matrix.npy"


def create_kernel_matrix():
    binary_list = []
    filename_list = []
    funcname_list = []
    train_funcnames= []
    Acfg_list = []
    funcname_large_list = []
    funcname_small_list = []
    paths = glob.glob(Acfg_path)

    ## collect all the binaries' cfg
    for path in paths:
        basename = os.path.basename(path).strip(".cfg")
        filename_list.append(basename)
        picklefile = open(path, "r")
        cfg = pickle.load((StrToBytes(picklefile)))
        picklefile.close()
        binary_list.append(cfg)
        del cfg

    ## collect all the funcname that nodes' num smaller than max_limit and larger than max_limit
    for i, binary in enumerate(binary_list):
        binary_name = filename_list[i]
        for func in binary.raw_graph_list:
            funcname = binary_name + "-" + func.funcname
            if len(func.g.nodes) > max_limit:
                funcname_large_list.append(funcname)
                continue
            if len(func.g.nodes) < min_limit:
                funcname_small_list.append(funcname)
                continue
            funcname_list.append(funcname)
            # Acfg_list.append(func.g)

    ## random select 1w function as the baseline dataset
    query,funcname_list,labels = random_sample(funcname_list)

    ## store the intermediate file
    np.save(query_path,query)
    np.save(labels_path,labels)
    funcname_large_file = open(large_func_path, "w")
    for funcname in funcname_large_list:
        funcname_large_file.write(funcname + "\n")
    funcname_large_file.close()
    small_func_file = open(small_func_path,"w")
    for funcname in funcname_small_list:
        small_func_file.write(funcname + "\n")
    small_func_file.close()


    func_num = len(funcname_list)
    kernel_matrix = np.zeros((func_num, func_num))

    for i, binary in enumerate(binary_list):
        binary_name = filename_list[i]
        for func in binary.raw_graph_list:
            funcname = binary_name + "-" + func.funcname
            if funcname in funcname_list:
                Acfg_list.append(func.g)
                train_funcnames.append(funcname)
    del binary_list
    gc.collect()

    funcname_list = train_funcnames
    funcanme_file = open(train_func_path, "w")
    for funcanme in funcname_list:
        funcanme_file.write(funcanme + "\n")
    funcanme_file.close()

    print(len(Acfg_list))
    arg_list = []
    for i in range(func_num):
        arg_list.append((i, func_num, Acfg_list))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, vector in zip(range(func_num), executor.map(compute_one_to_all, arg_list)):
            print("finished " + str(i) + " line", "   total:", func_num)
            kernel_matrix[i] = vector

    for i in range(func_num):
        for j in range(0,i):
            kernel_matrix[i,j] = kernel_matrix[j,i]
    return kernel_matrix


if __name__ == "__main__":
    # func1,func2 = load_two_functions()
    # weight = generate_weight(func1,func2)
    # similarity = bipartite_match(weight).compute_graph_similarity()
    start = time.time()
    kernel_matrix = create_kernel_matrix()
    end = time.time()
    print("the kernel matrix generation process's total time:", end - start)
    np.save(kernel_matrix_path, kernel_matrix)
    print("all finished")