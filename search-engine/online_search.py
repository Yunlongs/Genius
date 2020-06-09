from nearpy import Engine
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.distances import EuclideanDistance
from nearpy.distances import CosineDistance
from nearpy.hashes import RandomBinaryProjections
from nearpy.experiments import DistanceRatioExperiment
from redis import Redis
from nearpy.storage import RedisStorage
import pickle
import numpy as np
import os
import time


class LSH(object):

    def __init__(self):
        self.feature_list = {}
        self.engine = None

    def loadHashmap(self, feature_size=129, result_n=1000):  # 这里参数没有用到
        '''
        feature_size: hash空间维数大小
        result_n :返回多少个最近邻
        '''
        # Create redis storage adapter
        redis_object = Redis(host='localhost', port=6379, db=0)
        redis_storage = RedisStorage(redis_object)
        try:
            # Get hash config from redis
            config = redis_storage.load_hash_configuration('test')
            # Config is existing, create hash with None parameters
            lshash = RandomBinaryProjections(None, None)
            # Apply configuration loaded from redis
            lshash.apply_config(config)

        except:
            # Config is not existing, create hash from scratch, with 10 projections
            lshash = RandomBinaryProjections('test', 10)

        # Create engine for feature space of 100 dimensions and use our hash.
        # This will set the dimension of the lshash only the first time, not when
        # using the configuration loaded from redis. Use redis storage to store
        # buckets.
        nearest = NearestFilter(result_n)
        # self.engine = Engine(feature_size, lshashes=[], vector_filters=[])
        self.engine = Engine(feature_size, lshashes=[lshash], vector_filters=[nearest], storage=redis_storage,
                             distance=EuclideanDistance())

        # Do some stuff like indexing or querying with the engine...

        # Finally store hash configuration in redis for later use
        redis_storage.store_hash_configuration(lshash)

    def appendToDB(self, funcname, fvector):
        if fvector is None:
            return
        self.engine.store_vector(np.asarray(fvector),funcname)

    def batch_appendDB(self, features,):
        for funcname in features:
            feature = features[funcname]
            self.appendToDB(funcname, feature)



    def batch_appendDBbyDir1(self, base_dir):
        image_dir = os.path.join(base_dir, "image")
        firmware_featrues = {}
        bnum = 0
        fnum = 0
        i = 0
        pdb.set_trace()
        for firmware_name in os.listdir(image_dir):
            print(firmware_name)
            firmware_featrues[firmware_name] = {}
            firmware_dir = os.path.join(image_dir, firmware_name)
            for binary_name in os.listdir(firmware_dir):
                if binary_name.endswith(".features"):
                    bnum += 1
                    featrues_dir = os.path.join(firmware_dir, binary_name)
                    featrues = pickle.load(open(featrues_dir, "r"))
                    for funcname in featrues:
                        fnum += 1
                        # pdb.set_trace()
                        feature = featrues[funcname]
                        self.appendToDB(binary_name, funcname, feature, firmware_name)
                    del featrues
        print("bnum ", bnum)
        print("fnum ", fnum)

    def dump(self, base_dir):
        db_dir = os.path.join(base_dir, "data/db/busybox.feature_mapping")
        pickle.dump(self.feature_list, open(db_dir, 'w'))
        db_dir = os.path.join(base_dir, "data/db/busybox.hashmap")
        pickle.dump(self.engine, open(db_dir, 'w'))

    def loadDB(self, base_dir):
        db_dir = os.path.join(base_dir, "data/db/busybox.feature_mapping")
        self.feature_list = pickle.load(open(db_dir, 'r'))
        db_dir = os.path.join(base_dir, "data/db/busybox.hashmap")
        self.engine = pickle.load(open(db_dir, 'r'))

    def findF(self, binary_name, funcname):
        x = [v for v in self.feature_list if
             binary_name in self.feature_list[v] and funcname in self.feature_list[v][binary_name]]
        return x[0]

def retrieveFeatures(features_vlad):
    features = {}
    with open("output/train_func.txt","r") as f:
        for i,line in enumerate(f.readlines()):
            features[line.strip()] = features_vlad[i,:]
    query_feature_matrix = np.load("output/query_features.npy")
    with open("output/query_cfgs","rb") as f:
        query_names = pickle.load(f)
        query_names = list(query_names.keys())

    query_features = {}
    for i,query in enumerate(query_names):
        query_features[query] = query_feature_matrix[i,:]
    return features,query_features

def display(x):
    i=0
    for res in x:
        if i>10:
            break
        #print("features:",res[0])
        print("funcname:",res[1]," distance:",res[2])
        i+=1

def QueryByOne(features_vlad,query_name):
    db_distance = LSH()
    feature_size = features_vlad.shape[1]
    db_distance.loadHashmap(feature_size=feature_size,result_n=1000)
    features,query_features = retrieveFeatures(features_vlad)
    db_distance.batch_appendDB(features)

    query = query_features[query_name]
    x = db_distance.engine.neighbours(query)
    display(x)

def reslove_name(res_name):
    funcname = res_name[res_name.rindex("-")+1:]
    binary_name = res_name[:res_name.rindex("-")]
    return binary_name,funcname

def QueryByMany(features_vlad,queries_name,topk):
    db_distance = LSH()
    feature_size = features_vlad.shape[1]
    db_distance.loadHashmap(feature_size=feature_size, result_n=1000)
    features,query_features = retrieveFeatures(features_vlad)
    db_distance.batch_appendDB(features)

    hits = 0
    for query_name in queries_name:
        query = query_features[query_name]
        res = db_distance.engine.neighbours(query)
        if len(res)==0:
            continue
        _,query_funcname = reslove_name(query_name)
        hit = 0
        for i in range(topk):
            res_name = res[i][1]
            _,target_funcname = reslove_name(res_name)
            if query_funcname == target_funcname:
                hit +=1
            if i == len(res)-1:
                break
        acc = hit/topk
        hits +=hit
        print("for the "+str(i)+"st query: ",query_name," accuracy:",acc)
        display(res)
        print("-----------------------")
    avg_acc = hits/(topk*len(queries_name))
    print("total accury:",avg_acc)


def RandomSelectQueries(k):
    lines = np.load("output/query.npy")
    num = len(lines)
    np.random.seed(10)
    rand_num = np.random.randint(low=0,high=num,size=k)
    queries = []
    for i in range(k):
        queries.append(lines[rand_num[i]].strip())
    print("Sampled queries:")
    print(queries)
    return queries

def check_result(features_vlad,query_name,target_name,returned_name):
    features, query_features = retrieveFeatures(features_vlad)
    matrix = np.zeros((3,16))
    matrix[0,:] = query_features[query_name]
    matrix[1,:] = features[target_name]
    matrix[2,:] = features[returned_name]
    target_distance = np.sqrt(np.sum((matrix[0,:]-matrix[1,:])**2))
    return_distance = np.sqrt(np.sum((matrix[0,:]-matrix[2,:])**2))
    print("query&target distance:",target_distance)
    print("query&retrieve distance:",return_distance)
    print(123)



if __name__ == '__main__':
    features_vlad = np.load("output/feature_encoding.npy")
    #QueryByOne(features_vlad,"openssl-101a_x86_gcc_O2_openssl-level_find_node")
    #queries = RandomSelectQueries(10)
    #start = time.time()
    #QueryByMany(features_vlad,queries,50)
    #end = time.time()
    #print("total time:",end-start)
    check_result(features_vlad,"openssl-101a_x86_gcc_O2_openssl-level_find_node","openssl-101a_arm_clang_O0_openssl-level_find_node","openssl-101f_mips_clang_O0_openssl-DES_ede3_ofb64_encrypt")