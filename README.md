# Genius
 
## About
The project is an complete implementation of Genius, a framework for binary similarity detection.

The paper's full name is "Scalable Graph-based Bug Search for Firmware Images",you can download and read it from google.

I have written a note about this approach,you can see at [Scalable Graph-based Bug Search for Firmware Images(Genius)阅读笔记](http://yunlongs.cn/2020/03/20/Genius/)
 
The author had provided [a version of codes about Genius](https://github.com/qian-feng/Gencoding),but there are only two modules of Genius that we can's use it directly.

---
## Requirment
```buildoutcfg
IDA 7.0
networkx >2.3 in py3
redis
```

## How to run
### raw-feature-extractor

Set some arguments in `batch_run.py`,and you can run it for extracting the ACFG.
- IDA_PATH:where is your IDA located at.
- PLUGIN_PATH:where is the `preprocessing_ida` script located at.
- ELF_PATH: The executable files' path.
- DST_path: Where you would like to store the ACFG.

Please note,you must run `batch_run.py`**by python2**.Because the version of IDA only support python2 to write plugins.

### search-engine
**First**,you must run `train.py` to create kernel_matrix by bipartite graph matching algorithm.

**Second**,run the `feature_encoding.py` to encoding features for each function.

**Third**, run the `online_search.py` to do the search.

Please note,the scripts in this directory must run **by python3**.