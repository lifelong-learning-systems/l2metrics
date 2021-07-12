import json
from typing import Tuple
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
    
    def getBackwardTransferRatio(self,taska:str=None,taskb:str=None)->dict:
        if taska is None:
            return self.df.root.backward_transfer_ratio
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].backward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].backward_transfer_ratio[taskb].tolist()[0]

    def getForwardTransferRatio(self,taska:str=None,taskb:str=None)->dict:
        if taska is None:
            return self.df.root.forward_transfer_ratio
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].forward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].forward_transfer_ratio[taskb].tolist()[0]
    
    def getBackwardTransferContrast(self,taska:str=None,taskb:str=None)->dict:
        if taska is None:
            return self.df.root.backward_transfer_contrast
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].backward_transfer_contrast)
        else:
            return self.df.root.task_metrics[taska].backward_transfer_contrast[taskb].tolist()[0]

    def getForwardTransferContrast(self,taska:str=None,taskb:str=None)->dict:
        if taska is None:
            return self.df.root.forward_transfer_contrast
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].forward_transfer_contrast)
        else:
            return self.df.root.task_metrics[taska].forward_transfer_contrast[taskb].tolist()[0]

    def getMaintenanceValMRLEP(self,task:str)->list:
        return self.df.root.task_metrics[task].maintenance_val_mrlep.tolist()[0]
    
    def getMaintenanceValMRTLP(self,task:str)->list:
        return self.df.root.task_metrics[task].maintenance_val_mrtlp.tolist()[0]
    
    def getRecoveryTimes(self,task:str)->list:
        return self.df.root.task_metrics[task].recovery_times.tolist()[0]
    
    def getPerfRecoveryRate(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_recovery
        else:
            return self.df.root.task_metrics[task].perf_recovery
    
    def getPerfMaintenanceMRLEP(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_maintenance_mrlep
        else:
            return self.df.root.task_metrics[task].perf_maintenance_mrlep
    
    def getPerfMaintenanceMRtLP(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_maintenance_mrtlp
        else:
            return self.df.root.task_metrics[task].perf_maintenance_mrtlp
    
    def getSteRelPerf(self, task:str=None)->int:
        if task is None:
            return self.df.root.ste_rel_perf
        else:
            return self.df.root.task_metrics[task].ste_rel_perf
    
    def getSampleEfficiency(self, task:str=None)->int:
        if task is None:
            return self.df.root.sample_efficiency
        else:
            return self.df.root.task_metrics[task].sample_efficiency
    
    def getRunID(self)->int:
        return self.df.root.run_id
    
    def getComplexity(self)->int:
        return self.df.root.complexity

    def getDifficulty(self)->int:
        return self.df.root.difficulty

    def getScenarioType(self)->int:
        return self.df.root.scenario_type
    
    def getMetricsColumn(self)->int:
        return self.df.root.metrics_column

    def getMinMax(self,task:str=None)->Tuple[int,int]:
        if task is None:
            return self.df.root.min,self.df.root.max
        else:
            return self.df.root.task_metrics[task].min,self.df.root.task_metrics[task].max
    
    def getMinMax(self,task:str=None)->Tuple[int,int]:
        if task is None:
            return self.df.root.min,self.df.root.max
        else:
            return self.df.root.task_metrics[task].min,self.df.root.task_metrics[task].max