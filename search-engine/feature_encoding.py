from sklearn.cluster import SpectralClustering
import numpy as np
from train import *
import pickle

kernel_matrix = np.load("output/kernel_matrix.npy")
codebook_size = 16
query_funcnames = np.load("output/query.npy")

## spectral clustering
clustering = SpectralClustering(n_clusters=codebook_size,affinity='precomputed',assign_labels="kmeans")
labels = clustering.fit_predict(kernel_matrix)
clusters = {}
for node,label in enumerate(labels):
    if clusters.get(label) == None:
        clusters[label] = []
    clusters[label].append(node)

def compute_cluster_centroid(clusters,kernel_matrix):
    '''
    根据谱聚类的结果，寻找出每一类的聚类中心
    '''
    centroid = []
    for cluster in clusters.values():
        distance_list = []
        for node1 in cluster:
            distance = 0
            for node2 in cluster:
                if node1 == node2:
                    continue
                distance+=(1- kernel_matrix[node1,node2])
            distance_list.append(distance)
        distance_list = np.array(distance_list)
        c = np.argmin(distance_list)
        centroid.append(cluster[c])
    return centroid

def KNearestNeighbors(node,centroid,kernel_matrix,k=10):
    '''
    寻找当前样本，距离最近的10个聚类中心
    '''
    distance = []
    for c in centroid:
        distance.append(1-kernel_matrix[node,c])
    min_distance = np.array(distance).argsort()
    kneighbors = []
    for i in range(k):
        kneighbors.append(min_distance[i])
    return kneighbors

def vlad_encoding(kernel_matrix,centroid,k=10):
    node_numbers = kernel_matrix.shape[0]
    feature_matrix = np.zeros((node_numbers,codebook_size))
    for node in range(node_numbers):
        kneighbors = KNearestNeighbors(node,centroid,kernel_matrix,k)
        for c_id in kneighbors:
            feature_matrix[node,c_id] = kernel_matrix[node,centroid[c_id]]
    return feature_matrix

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def load_centroid(centroid):
    '''
    获得聚类中心样本点所对应的函数名
    '''
    func_list = []
    with open("output/train_func.txt") as f:
        for line in f.readlines():
            func_list.append(line.strip())
    centroid_name = []
    for c in centroid:
        centroid_name.append(func_list[c])
    print(centroid_name)
    return centroid_name

def load_queries(funcnames,centroid):
    """
    如果在train时没有保存query和centroid对应的cfg，那么就需要根据保存的函数名，从cfg数据库中寻找这些cfg
    @funcname:Queries的函数名列表
    @centroid:聚类中心的列表
    """
    funcnames = np.sort(funcnames)
    centroid_names = load_centroid(centroid)
    centroid_feature = {}
    features = {}
    func_list =[]
    last =""
    for i,funcname in enumerate(funcnames):
        filename = funcname[:funcname.rindex('-')]+".cfg"
        func = funcname[funcname.rindex('-')+1:]
        func_list.append(func)
        if last =="":
            last = filename
        if last !="" and (last != filename or i == len(funcnames)-1):
            if i==len(funcnames)-1:
                last = filename
            else:
                func_list.pop(-1)
            picklefile = open("../raw-feature-extractor/extracted-acfg/"+last,'r')
            cfg = pickle.load(StrToBytes(picklefile))
            picklefile.close()
            graph_list = cfg.raw_graph_list
            for g in graph_list:
                for centroid_name in centroid_names:
                    c_filename = centroid_name[:centroid_name.rindex('-')]+".cfg"
                    c_funcname = centroid_name[centroid_name.rindex('-')+1:]
                    if c_filename ==last and  g.funcname == c_funcname:
                        centroid_feature[centroid_name] = g
                for f in func_list:
                    if g.funcname==f:
                        features[last.strip(".cfg")+'-'+f] = g
            last=filename
            func_list = []
            func_list.append(func)
            print("finish:"+str(i))
    term ={}
    for name in centroid_names:
        term[name] = centroid_feature[name]
    centroid_feature =term

    with open("output/query_cfgs","wb") as f:
        pickle.dump(features,f)

    with open("output/centroid_cfgs","wb") as f:
        pickle.dump(centroid_feature,f)

    return features,centroid_feature



def vlad_encoding_for_queries(k=10):
    f = open("output/query_cfgs", "rb")
    query_cfgs = pickle.load(f)
    f.close()
    f = open("output/centroid_cfgs", "rb")
    centroid_cfgs = pickle.load(f)
    f.close()

    x,y = len(query_cfgs),len(centroid_cfgs)
    matrix = np.zeros((x,y))
    for i,query in enumerate(query_cfgs.values()):
        for j,c in enumerate(centroid_cfgs.values()):
            weight = generate_weight(query.g,c.g)
            similarity = bipartite_match(weight).compute_graph_similarity()
            matrix[i,j]=similarity
    np.save("output/query&centroid_matrix.npy",matrix)

    centroid = [i for i in range(codebook_size)]
    node_numbers = matrix.shape[0]
    feature_matrix = np.zeros((node_numbers, codebook_size))
    for node in range(node_numbers):
        kneighbors = KNearestNeighbors(node, centroid, matrix, k)
        for c_id in kneighbors:
            feature_matrix[node, c_id] = matrix[node, c_id]
    np.save("output/query_features.npy",feature_matrix)
    return feature_matrix


if __name__ == '__main__':
    centroid = compute_cluster_centroid(clusters,kernel_matrix)
    feature_matrix = vlad_encoding(kernel_matrix,centroid)
    np.save("output/feature_encoding.npy",feature_matrix)
    _,_ = load_queries(query_funcnames,centroid)
    _ = vlad_encoding_for_queries(k=10)




