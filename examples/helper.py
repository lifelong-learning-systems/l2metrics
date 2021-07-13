import json
import pprint
from typing import Tuple, Union
import pandas as pd
import seaborn as sns
from functools import reduce

class TaskMetrics:
    def __init__(self, json_file_name):
        with open(json_file_name) as file:
            data = json.load(file)
        self.df = pd.DataFrame(self.json_refactor(data[0] ))
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
                # elif a[key] == b[key]:
                #     pass # same leaf value
                # else:

                    # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a
    def merge_dict(self,dict1, dict2):
        for key, val in dict1.items():
            if type(val) == dict:
                if key in dict2 and type(dict2[key] == dict):
                    self.merge_dict(dict1[key], dict2[key])
            else:
                if key in dict2:
                    dict1[key] = dict2[key]

        for key, val in dict2.items():
            if not key in dict1:
                dict1[key] = val

        return dict1
    
    def df2dict_helper(self,key:list,val:any):
        new_dict={}
        if len(key) == 1:
            new_dict[key[0]] = val[0]
        else:
            new_dict[key[0]] = self.df2dict_helper(key[1:],val)
        return new_dict
    
    def df2dict(self,df:pd.DataFrame)->dict:
        pre_new_dict = []
        # print(df.columns)
        for col in df.columns:
            val = df[col]
            newcol = [x for x in list(col) if x==x] if isinstance(col,tuple) else [col]
            if newcol[0] == 'root':
                newcol.pop(0)
            pre_new_dict.append(self.df2dict_helper(newcol,val))
        # pprint.pprint(pre_new_dict)
        return reduce(self.mergedict,pre_new_dict,)
    
    def getNormalizationDataRange(self,task:str=None)->Union[dict,Tuple[int,int]]:
        if task is None:
            return self.df2dict(self.df.root.normalization_data_range)
        else:
            return self.df.root.normalization_data_range[task]["min"],self.df.root.normalization_data_range[task]["max"]

    def getBackwardTransferRatio(self,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        if taska is None:
            return self.df.root.backward_transfer_ratio.iloc[0,0]
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].backward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].backward_transfer_ratio[taskb].tolist()[0]

    def getForwardTransferRatio(self,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        if taska is None:
            return self.df.root.forward_transfer_ratio.iloc[0,0]
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].forward_transfer_ratio)
        else:
            return self.df.root.task_metrics[taska].forward_transfer_ratio[taskb].tolist()[0]
    
    def getBackwardTransferContrast(self,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        if taska is None:
            return self.df.root.backward_transfer_contrast.iloc[0,0]
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].backward_transfer_contrast)
        else:
            return self.df.root.task_metrics[taska].backward_transfer_contrast[taskb].tolist()[0]

    def getForwardTransferContrast(self,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        if taska is None:
            return self.df.root.forward_transfer_contrast.iloc[0,0]
        elif taskb is None:
            return self.df2dict(self.df.root.task_metrics[taska].forward_transfer_contrast)
        else:
            return self.df.root.task_metrics[taska].forward_transfer_contrast[taskb].tolist()[0]

    def getMaintenanceValMRLEP(self,task:str)->list:
        return self.df.root.task_metrics[task].maintenance_val_mrlep.iloc[0,0]
    
    def getMaintenanceValMRTLP(self,task:str)->list:
        return self.df.root.task_metrics[task].maintenance_val_mrtlp.iloc[0,0]
    
    def getRecoveryTimes(self,task:str)->list:
        return self.df.root.task_metrics[task].recovery_times.iloc[0,0]
    
    def getPerfRecoveryRate(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_recovery.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].perf_recovery.iloc[0,0]
    
    def getPerfMaintenanceMRLEP(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_maintenance_mrlep.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].perf_maintenance_mrlep.iloc[0,0]
    
    def getPerfMaintenanceMRTLP(self,task:str=None)->int:
        if task is None:
            return self.df.root.perf_maintenance_mrtlp.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].perf_maintenance_mrtlp.iloc[0,0]
    
    def getSTERelPerf(self, task:str=None)->int:
        if task is None:
            return self.df.root.ste_rel_perf.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].ste_rel_perf.iloc[0,0]
    
    def getSampleEfficiency(self, task:str=None)->int:
        if task is None:
            return self.df.root.sample_efficiency.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].sample_efficiency.iloc[0,0]
    
    def getRunID(self)->str:
        return self.df.root.run_id.iloc[0,0]
    
    def getComplexity(self)->str:
        return self.df.root.complexity.iloc[0,0]

    def getDifficulty(self)->str:
        return self.df.root.difficulty.iloc[0,0]

    def getScenarioType(self)->str:
        return self.df.root.scenario_type.iloc[0,0]
    
    def getMetricsColumn(self)->str:
        return self.df.root.metrics_column.iloc[0,0]

    def getMinMax(self,task:str=None)->Tuple[int,int]:
        if task is None:
            return self.df.root["min"].iloc[0,0],self.df.root["max"].iloc[0,0]
        else:
            return self.df.root.task_metrics[task]["min"],self.df.root.task_metrics[task]["max"]
    
    def getNumLXEX(self,task:str=None)->Tuple[int,int]:
        if task is None:
            return self.df.root.num_lx.iloc[0,0],self.df.root.num_ex.iloc[0,0]
        else:
            return self.df.root.task_metrics[task].num_lx.iloc[0,0],self.df.root.task_metrics[task].num_ex.iloc[0,0]