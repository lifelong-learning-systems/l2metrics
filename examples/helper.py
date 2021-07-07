import json
import pandas as pd
import seaborn as sns
from functools import reduce

class TaskMetrics:
    def __init__(self, json_file_name):
        with open(json_file_name) as file:
            data = json.load(file)
        self.df = pd.DataFrame(self.json_refactor(data))
        pass

    def json_refactor(self,json:dict,parent:str="root")->dict:
        new_dict={}
        for k,v in json.items():
            # print(parent,k,v)
            if isinstance(v,dict):
                for k1,v1 in self.json_refactor(v,parent=k).items():
                    # print(parent,k1,v1)
                    if isinstance(k1,tuple):
                        # print("1", type(tuple((parent))),type(k1),tuple([parent])+k1)
                        # print("1",sum((tuple((parent)), k1),()))
                        if isinstance(v1,list)  and len(v1)!=1:
                            new_dict[tuple([parent])+k1]=[v1]#pd.Series(v1)
                        else:
                            new_dict[tuple([parent])+k1]=v1
                    else:
                        # print("2",(parent,k1))
                        if isinstance(v1,list) and len(v1)!=1:
                            new_dict[(parent,k1)]=[v1]#pd.Series(v1)
                        else:
                            new_dict[(parent,k1)]=v1 
            else:
                # print("3",(parent,k))
                new_dict[(parent,k)]=v
        return new_dict
    
    # sourced from stackoverflow
    # link: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    def mergedict(self,a, b, path=None):
        if path is None: path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    self.mergedict(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass # same leaf value
                else:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a
    
    def df2dict_helper(self,key:list,val:any):
        new_dict={}
        if len(key) == 1:
            new_dict[key[0]] = val[0]
        else:
            new_dict[key[0]] = self.df2dict_helper(key[1:],val)
        return new_dict
    
    def df2dict(self,df:pd.DataFrame)->dict:
        pre_new_dict = []
        # if isinstance(df,pd.Series):
        #     return {parent:df.tolist()[0]}
        for col in df.columns:
            val = df[col]
            newcol=[x for x in list(col) if x==x][1:] if isinstance(col,tuple) else [col]
            pre_new_dict.append(self.df2dict_helper(newcol,val))

        return reduce(self.mergedict,pre_new_dict)
    
    def getBackwardTransferRatio(self,taska:str,taskb:str=None)->dict:
        if taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].backward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].backward_transfer_ratio[taskb].tolist()[0]

    def getForwardTransferRatio(self,taska:str,taskb:str=None)->dict:
        if taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].forward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].forward_transfer_ratio[taskb].tolist()[0]
    
    def getPerfRecoveryRate(self,task:str=None):
        if task is None:
            return self.df2dict(self.df.root.perf_recovery)
        else:
            return self.df2dict(self.df.root.task_metrics[task].perf_recovery)