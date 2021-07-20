import json
import pprint
from typing import List, Tuple, Union
import pandas as pd
import seaborn as sns
from functools import reduce

'''
TODOS:
    [ ] refactor to 
'''

class TaskMetrics:
    def __init__(self, json_file_name):
        with open(json_file_name) as file:
            self.data = json.load(file)
        self.dfs = [pd.DataFrame(self.json_refactor(run_num)) for run_num in self.data]
        # pass

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
    # def merge_dict(self,dict1, dict2):
    #     for key, val in dict1.items():
    #         if type(val) == dict:
    #             if key in dict2 and type(dict2[key] == dict):
    #                 self.merge_dict(dict1[key], dict2[key])
    #         else:
    #             if key in dict2:
    #                 dict1[key] = dict2[key]

    #     for key, val in dict2.items():
    #         if not key in dict1:
    #             dict1[key] = val

    #     return dict1
    
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
    
    def getNormalizationDataRange_helper(self,df:pd.DataFrame,task:str=None)->Union[dict,Tuple[int,int]]:
        # print(type(df),df.root)
        try:
            if task is None:
                return self.df2dict(df.root.normalization_data_range)
            else:
                return df.root.normalization_data_range[task]["min"].iloc[0,0],df.root.normalization_data_range[task]["max"].iloc[0,0]
        except KeyError:
            pass

    def getNormalizationDataRange(self,task:str=None)->List[Union[dict,Tuple[int,int]]]:
        return [self.getNormalizationDataRange_helper(run,task) for run in self.dfs]

    def getBackwardTransferRatio_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.backward_transfer_ratio.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].backward_transfer_ratio)
            else:
                return df.root.task_metrics[taska].backward_transfer_ratio[taskb].tolist()[0]
        except KeyError:
            pass

    def getBackwardTransferRatio(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getBackwardTransferRatio_helper(run,taska,taskb) for run in self.dfs]

    def getForwardTransferRatio_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.forward_transfer_ratio.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].forward_transfer_ratio)
            else:
                return df.root.task_metrics[taska].forward_transfer_ratio[taskb].tolist()[0]
        except KeyError:
            pass

    def getForwardTransferRatio(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getForwardTransferRatio_helper(run,taska,taskb) for run in self.dfs]

    def getBackwardTransferContrast_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.backward_transfer_contrast.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].backward_transfer_contrast)
            else:
                return df.root.task_metrics[taska].backward_transfer_contrast[taskb].tolist()[0]
        except KeyError:
            pass

    def getBackwardTransferContrast(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getBackwardTransferContrast_helper(run,taska,taskb) for run in self.dfs]

    def getForwardTransferContrast_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.forward_transfer_contrast.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].forward_transfer_contrast)
            else:
                return df.root.task_metrics[taska].forward_transfer_contrast[taskb].tolist()[0]
        except KeyError:
            pass
    def getForwardTransferContrast(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getForwardTransferContrast_helper(run,taska,taskb) for run in self.dfs]

    def getMaintenanceValMRLEP_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].maintenance_val_mrlep.iloc[0,0]
        except KeyError:
            pass
    
    def getMaintenanceValMRLEP(self,task:str)->List[list]:
        return list(filter(None,[self.getMaintenanceValMRLEP_helper(run,task) for run in self.dfs]))
    
    def getMaintenanceValMRTLP_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].maintenance_val_mrtlp.iloc[0,0]
        except KeyError:
            pass
    
    def getMaintenanceValMRTLP(self,task:str)->List[list]:
        return [self.getMaintenanceValMRTLP_helper(run,task) for run in self.dfs]
    
    def getRecoveryTimes_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].recovery_times.iloc[0,0]
        except KeyError:
            pass
    
    def getRecoveryTimes(self,task:str=None)->List[list]:
        return  [self.getRecoveryTimes_helper(run,task) for run in self.dfs] 
    
    def getPerfRecoveryRate_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_recovery.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_recovery.iloc[0,0]
        except KeyError:
            pass
    
    def getPerfRecoveryRate(self,task:str=None)->List[int]:
        return [self.getPerfRecoveryRate_helper(run,task) for run in self.dfs]
    
    def getPerfMaintenanceMRLEP_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_maintenance_mrlep.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrlep.iloc[0,0]
        except KeyError:
            pass
    
    def getPerfMaintenanceMRLEP(self,task:str=None)->List[int]:
        return [self.getPerfMaintenanceMRLEP_helper(run,task) for run in self.dfs]
    
    def getPerfMaintenanceMRTLP_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_maintenance_mrtlp.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrtlp.iloc[0,0]
        except KeyError:
            pass
    
    def getPerfMaintenanceMRTLP(self,task:str=None)->List[int]:
        return [self.getPerfMaintenanceMRTLP_helper(run,task) for run in self.dfs]
    
    def getSTERelPerf_helper(self,df:pd.DataFrame, task:str=None)->int:
        try:
            if task is None:
                return df.root.ste_rel_perf.iloc[0,0]
            else:
                return df.root.task_metrics[task].ste_rel_perf.iloc[0,0]
        except KeyError:
            pass

    def getSTERelPerf(self,task:str=None)->List[int]:
        return [self.getSTERelPerf_helper(run,task) for run in self.dfs]
    
    
    def getSampleEfficiency_helper(self,df:pd.DataFrame, task:str=None)->int:
        try:
            if task is None:
                return df.root.sample_efficiency.iloc[0,0]
            else:
                return df.root.task_metrics[task].sample_efficiency.iloc[0,0]
        except KeyError:
            pass
    
    def getSampleEfficiency(self,task:str=None)->List[int]:
        return [self.getSampleEfficiency_helper(run,task) for run in self.dfs]

    def getRunID_helper(self,df:pd.DataFrame)->str:
        return df.root.run_id.iloc[0,0]
    
    def getRunID(self)->List[str]:
        return [self.getRunID_helper(run) for run in self.dfs]
    
    def getComplexity_helper(self,df:pd.DataFrame)->str:
        return df.root.complexity.iloc[0,0]
    
    def getComplexity(self)->List[str]:
        return [self.getComplexity_helper(run) for run in self.dfs]

    def getDifficulty_helper(self,df:pd.DataFrame)->str:
        return df.root.difficulty.iloc[0,0]
    
    def getDifficulty(self,)->List[str]:
        return [self.getDifficulty_helper(run) for run in self.dfs]

    def getScenarioType_helper(self,df:pd.DataFrame)->str:
        return df.root.scenario_type.iloc[0,0]
    
    def getScenarioType(self)->List[str]:
        return [self.getScenarioType_helper(run) for run in self.dfs]
    
    def getMetricsColumn_helper(self,df:pd.DataFrame)->str:
        return df.root.metrics_column.iloc[0,0]
    
    def getMetricsColumn(self)->List[str]:
        return [self.getMetricsColumn_helper(run) for run in self.dfs]

    def getMinMax_helper(self,df:pd.DataFrame,task:str=None)->Tuple[int,int]:
        try:
            if task is None:
                return df.root["min"].iloc[0,0],df.root["max"].iloc[0,0]
            else:
                return df.root.task_metrics[task]["min"],df.root.task_metrics[task]["max"]
        except KeyError:
            pass
    
    def getMinMax(self,task:str=None)->List[Tuple[int,int]]:
        return [self.getMinMax_helper(run,task) for run in self.dfs]
    
    def getNumLXEX_helper(self,df:pd.DataFrame,task:str=None)->Tuple[int,int]:
        try:
            if task is None:
                return df.root.num_lx.iloc[0,0],df.root.num_ex.iloc[0,0]
            else:
                return df.root.task_metrics[task].num_lx.iloc[0,0],df.root.task_metrics[task].num_ex.iloc[0,0]
        except KeyError:
            pass

    def getNumLXEX(self,task:str=None)->List[Tuple[int,int]]:
        return [self.getNumLXEX_helper(run,task) for run in self.dfs]