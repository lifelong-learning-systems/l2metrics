import json
import pprint
from typing import List, Tuple, Union
import pandas as pd
import seaborn as sns
import matplotlib as plt
from functools import reduce

class MetricsJsonParser:
    def __init__(self, json_file_name):
        with open(json_file_name) as file:
            self.data = json.load(file)
        self.dfs = [pd.DataFrame(self.json_refactor(run_num)) for run_num in self.data]
        # pass
    
    def toXL(self,):
        for idx,df in enumerate(self.dfs):
            df.to_excel(str(idx)+'.xlsx', header=True)
    
    def flatten(self,l:list):
        flat = []
        for subl in l:
            if subl:
                for item in subl:
                    flat.append(item)
            else:
                flat.append(None)
        return flat
        # return [item for subl in l if subl for item in subl else None]
    
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

    def plotLine(self,data:list): 
        sns.lineplot(x=range(len(data)),y=data)
    
    def plotHist(self,data:list): 
        sns.histplot(data)

    def plotDist(self,data:list): 
        sns.distplot(data)
    
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
        except (KeyError,AttributeError) as e:
            pass

    def getNormalizationDataRange(self,task:str=None)->List[Union[dict,Tuple[int,int]]]:
        return [self.getNormalizationDataRange_helper(run,task) for run in self.dfs]

    # possible types: 'hist','dist','line'
    def plotNormalizationDataRange(self,plottype:str,task:str=None):
        if task:
        # normdatrange = self.flatten(self.getNormalizationDataRange(task))
            normdatrange = [x for x in self.getNormalizationDataRange(task) if x]
            fig, ax = plt.pyplot.subplots(1,2,figsize=(18,10))
            if plottype == 'hist':
                sns.histplot([min for min,_ in normdatrange], ax=ax[0])
                sns.histplot([max for _,max in normdatrange], ax=ax[1])
            elif plottype == 'dist':
                sns.distplot([min for min,_ in normdatrange], ax=ax[0])
                sns.distplot([max for _,max in normdatrange], ax=ax[1])
            elif plottype == 'line':
                sns.lineplot([min for min,_ in normdatrange], ax=ax[0])
                sns.lineplot([max for _,max in normdatrange], ax=ax[1])
            return fig
        else:
            normdatrange = [x for x in self.getNormalizationDataRange(task) if x]
            graph_titles = set()
            for d in normdatrange:
                graph_titles.update(list(d.keys()))
            # print(graph_titles)
            graph_data = {k:[] for k in graph_titles}
            # print(graph_data)
            for d in normdatrange:
                for k in d:
                    # print(k)
                    graph_data[k].append((d[k]["min"],d[k]["max"]))
            # print(graph_data)
            fig, ax = plt.pyplot.subplots(len(graph_titles),2,figsize=(18,30))
            if plottype == 'hist':
                i=0
                for k,v in graph_data.items():
                    sns.histplot([min for min,_ in v], ax=ax[i][0]).set(title=k+' Min')
                    sns.histplot([max for _,max in v], ax=ax[i][1]).set(title=k+' Max')
                    i+=1
            elif plottype == 'dist':
                i=0
                for k,v in graph_data.items():
                    sns.distplot([min for min,_ in v], ax=ax[i][0]).set(title=k+' Min')
                    sns.distplot([max for _,max in v], ax=ax[i][1]).set(title=k+' Max')
                    i+=1
            elif plottype == 'line':
                i=0
                for k,v in graph_data.items():
                    sns.lineplot([min for min,_ in v], ax=ax[i][0]).set(title=k+' Min')
                    sns.lineplot([max for _,max in v], ax=ax[i][1]).set(title=k+' Max')
                    i+=1
            return fig

    def getBackwardTransferRatio_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.backward_transfer_ratio.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].backward_transfer_ratio)
            else:
                return df.root.task_metrics[taska].backward_transfer_ratio[taskb].tolist()[0]
        except (KeyError,AttributeError) as e:
            pass

    def getBackwardTransferRatio(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getBackwardTransferRatio_helper(run,taska,taskb) for run in self.dfs]

    def plotBackwardTransferRatio(self,plottype:str,taska:str=None,taskb:str=None):
        backtransfratio = self.getBackwardTransferRatio(taska,taskb)
        if not (taska and taskb):
            print("resulting query is in the form of dictionaries\ntherefore there will be no plot but instead a list of dict returned")
            return backtransfratio
        if isinstance( backtransfratio[0],list):
            reduce(lambda l,m: l+m if m else l+[m],backtransfratio)
        fig,ax = plt.pyplot.subplots(figsize=(10,5))
        if plottype == 'hist':
            sns.histplot(backtransfratio,ax=ax)
        elif plottype == 'dist':
            sns.distplot(backtransfratio,ax=ax)
        elif plottype == 'line':
            sns.lineplot(backtransfratio,ax=ax)
        return fig

    def getForwardTransferRatio_helper(self,df:pd.DataFrame,taska:str=None,taskb:str=None)->Union[int,dict,list]:
        try:
            if taska is None:
                return df.root.forward_transfer_ratio.iloc[0,0]
            elif taskb is None:
                return self.df2dict(df.root.task_metrics[taska].forward_transfer_ratio)
            else:
                return df.root.task_metrics[taska].forward_transfer_ratio[taskb].tolist()[0]
        except (KeyError,AttributeError) as e:
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
        except (KeyError,AttributeError) as e:
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
        except (KeyError,AttributeError) as e:
            pass
    def getForwardTransferContrast(self,taska:str=None,taskb:str=None)->List[Union[int,dict,list]]:
        return [self.getForwardTransferContrast_helper(run,taska,taskb) for run in self.dfs]

    def getMaintenanceValMRLEP_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].maintenance_val_mrlep.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getMaintenanceValMRLEP(self,task:str)->List[list]:
        return list(filter(None,[self.getMaintenanceValMRLEP_helper(run,task) for run in self.dfs]))
    
    def getMaintenanceValMRTLP_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].maintenance_val_mrtlp.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getMaintenanceValMRTLP(self,task:str)->List[list]:
        return [self.getMaintenanceValMRTLP_helper(run,task) for run in self.dfs]
    
    def getRecoveryTimes_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].recovery_times.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getRecoveryTimes(self,task:str=None)->List[list]:
        return  [self.getRecoveryTimes_helper(run,task) for run in self.dfs] 
    
    def getPerfRecoveryRate_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_recovery.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_recovery.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getPerfRecoveryRate(self,task:str=None)->List[int]:
        return [self.getPerfRecoveryRate_helper(run,task) for run in self.dfs]
    
    def getPerfMaintenanceMRLEP_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_maintenance_mrlep.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrlep.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getPerfMaintenanceMRLEP(self,task:str=None)->List[int]:
        return [self.getPerfMaintenanceMRLEP_helper(run,task) for run in self.dfs]
    
    def getPerfMaintenanceMRTLP_helper(self,df:pd.DataFrame,task:str=None)->int:
        try:
            if task is None:
                return df.root.perf_maintenance_mrtlp.iloc[0,0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrtlp.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getPerfMaintenanceMRTLP(self,task:str=None)->List[int]:
        return [self.getPerfMaintenanceMRTLP_helper(run,task) for run in self.dfs]
    
    def getSTERelPerf_helper(self,df:pd.DataFrame, task:str=None)->int:
        try:
            if task is None:
                return df.root.ste_rel_perf.iloc[0,0]
            else:
                return df.root.task_metrics[task].ste_rel_perf.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getSTERelPerf(self,task:str=None)->List[int]:
        return [self.getSTERelPerf_helper(run,task) for run in self.dfs]
    
    
    def getSampleEfficiency_helper(self,df:pd.DataFrame, task:str=None)->int:
        try:
            if task is None:
                return df.root.sample_efficiency.iloc[0,0]
            else:
                return df.root.task_metrics[task].sample_efficiency.iloc[0,0]
        except (KeyError,AttributeError) as e:
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
        except (KeyError,AttributeError) as e:
            pass
    
    def getMinMax(self,task:str=None)->List[Tuple[int,int]]:
        return [self.getMinMax_helper(run,task) for run in self.dfs]
    
    def getNumLXEX_helper(self,df:pd.DataFrame,task:str=None)->Tuple[int,int]:
        try:
            if task is None:
                return df.root.num_lx.iloc[0,0],df.root.num_ex.iloc[0,0]
            else:
                return df.root.task_metrics[task].num_lx.iloc[0,0],df.root.task_metrics[task].num_ex.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getNumLXEX(self,task:str=None)->List[Tuple[int,int]]:
        return [self.getNumLXEX_helper(run,task) for run in self.dfs]

    def getSTERelPerf_helper(self,df:pd.DataFrame,task:str=None):
        try:
            if task is None:
                return df.root.ste_rel_perf.iloc[0,0]
            else:
                return df.root.task_metrics[task].ste_rel_perf.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getSTERelPerf(self,task:str=None):
        return [self.getSTERelPerf_helper(run,task) for run in self.dfs]

    def getSampleEfficiency_helper(self,df:pd.DataFrame,task:str=None):
        try:
            if task is None:
                return df.root.sample_efficiency.iloc[0,0]
            else:
                return df.root.task_metrics[task].sample_efficiency.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
        
    def getSTERelPerf(self,task:str=None):
        return [self.getSampleEfficiency(run,task) for run in self.dfs]

    def getSTERelPerfVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].ste_rel_perf_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSTERelPerfVals(self,task:str)->List[list]:
        return [self.getSTERelPerfVals_helper(run,task) for run in self.dfs]
    
    def getSTESatVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].ste_saturation_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSTESatVals(self,task:str)->List[list]:
        return [self.getSTESatVals_helper(run,task) for run in self.dfs]
    
    def getSTEEps2SatVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].ste_eps_to_sat_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSTEEps2SatVals(self,task:str)->List[list]:
        return [self.getSTEEps2SatVals_helper(run,task) for run in self.dfs]
    
    def getSESatVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].se_saturation_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSESatVals(self,task:str)->List[list]:
        return [self.getSESatVals_helper(run,task) for run in self.dfs]
    
    def getSEEps2SatVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].se_eps_to_sat_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSEEps2SatVals(self,task:str)->List[list]:
        return [self.getSEEps2SatVals_helper(run,task) for run in self.dfs]

    def getSampleEfficiencyVals_helper(self,df:pd.DataFrame,task:str)->list:
        try:
            return df.root.task_metrics[task].sample_efficiency_vals.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass
    
    def getSampleEfficiencyVals(self,task:str)->List[list]:
        return [self.getSampleEfficiencyVals_helper(run,task) for run in self.dfs]

    def getSETaskSat_helper(self,df:pd.DataFrame,task:str):
        try:
            return df.root.task_metrics[task].se_task_saturation.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getSETaskSat(self,task:str):
        return [self.getSETaskSat_helper(run,task) for run in self.dfs]

    def getSETaskEPS2Sat_helper(self,df:pd.DataFrame,task:str):
        try:
            return df.root.task_metrics[task].se_task_eps_to_sat.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getSETaskEPS2Sat(self,task:str):
        return [self.getSETaskEPS2Sat_helper(run,task) for run in self.dfs]
    
    def getRuntime_helper(self,df:pd.DataFrame):
        try:
            return df.root.runtime.iloc[0,0]
        except (KeyError,AttributeError) as e:
            pass

    def getRuntime(self):
        return [self.getRuntime_helper(run) for run in self.dfs]