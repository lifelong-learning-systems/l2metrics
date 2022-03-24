"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
from functools import reduce
from typing import List, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MetricsParser:
    dfs = []
    df_tsv = None

    def __init__(self, file_name, tsv: bool = False) -> None:
        if tsv:
            self.df_tsv = pd.read_csv(file_name, sep="\t")
        else:
            with open(file_name) as file:
                self.data = json.load(file)
            self.dfs = [
                pd.DataFrame(self.refactor_json(run_num)).fillna(value=np.nan)
                for run_num in self.data
            ]

    def to_excel(self) -> None:
        for idx, df in enumerate(self.dfs):
            df.to_excel(str(idx) + ".xlsx", header=True)

    def flatten_list(self, l: list) -> List:
        flat = []
        for subl in l:
            if isinstance(subl, list):
                flat += subl
            elif isinstance(subl, float) or isinstance(subl, int):
                flat.append(subl)
            else:
                flat.append(None)
        return flat
        # return [item for subl in l if subl for item in subl else None]

    def flatten_dict(self, d: list) -> Tuple[Set, dict]:
        graph_titles = set()
        for d1 in d:
            if d1:
                graph_titles.update(list(d1.keys()))
        graph_data = {k: [] for k in graph_titles}
        for d1 in d:
            if d1:
                for k in d1:
                    if isinstance(d1[k], dict):
                        graph_data[k].append(list(d1[k].values()))
                    elif isinstance(d1[k], list):
                        graph_data[k] += d1[k]
                    else:
                        graph_data[k].append(d1[k])
        return graph_titles, graph_data

    def refactor_json(self, json: dict, parent: str = "root") -> dict:
        new_dict = {}
        for k, v in json.items():
            if isinstance(v, dict):
                for k1, v1 in self.refactor_json(v, parent=k).items():
                    if isinstance(k1, tuple):
                        if isinstance(v1, list) and len(v1) != 1:
                            new_dict[tuple([parent]) + k1] = [v1]
                        else:
                            new_dict[tuple([parent]) + k1] = v1
                    else:
                        if isinstance(v1, list) and len(v1) != 1:
                            new_dict[(parent, k1)] = [v1]
                        else:
                            new_dict[(parent, k1)] = v1
            else:
                new_dict[(parent, k)] = v
        return new_dict

    # sourced from stackoverflow
    # link: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    def merge_dict(self, a, b, path=None) -> dict:
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    self.merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        return a

    ##################################################
    # JSON methods
    ##################################################

    def _df2dict_helper(self, key: list, val) -> dict:
        new_dict = {}
        if len(key) == 1:
            new_dict[key[0]] = val[0]
        else:
            new_dict[key[0]] = self._df2dict_helper(key[1:], val)
        return new_dict

    def df2dict(self, df: pd.DataFrame) -> dict:
        pre_new_dict = []
        for col in df.columns:
            val = df[col]
            newcol = (
                [x for x in list(col) if x == x] if isinstance(col, tuple) else [col]
            )
            if newcol[0] == "root":
                newcol.pop(0)
            pre_new_dict.append(self._df2dict_helper(newcol, val))
        return reduce(
            self.merge_dict,
            pre_new_dict,
        )

    ##################################################
    # Normalization Data Range
    ##################################################

    def _get_normalization_data_range_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, dict, Tuple[int, int]]:
        try:
            if task:
                return (
                    df.root.normalization_data_range[task]["min"].iloc[0, 0],
                    df.root.normalization_data_range[task]["max"].iloc[0, 0],
                )
            else:
                return self.df2dict(df.root.normalization_data_range)
        except (KeyError, AttributeError):
            pass

    def get_normalization_data_range(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, dict, Tuple[int, int]]]:
        if not run_id:
            return [
                self._get_normalization_data_range_helper(run, task) for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_normalization_data_range_helper(run, task)

    # possible types: 'hist','dist','line'
    def plot_normalization_data_range(self, plottype: str, task: str = "") -> None:
        if task:
            # normdatrange = self.flatten(self.getNormalizationDataRange(task))
            normdatrange = [x for x in self.get_normalization_data_range(task) if x]
            _, axes = plt.subplots(1, 2, figsize=(12, 10), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([min for min, _ in normdatrange], ax=axes[0]).set(
                    title="Min"
                )
                sns.histplot([max for _, max in normdatrange], ax=axes[1]).set(
                    title="Max"
                )
            elif plottype == "dist":
                sns.distplot([min for min, _ in normdatrange], ax=axes[0]).set(
                    title="Min"
                )
                sns.distplot([max for _, max in normdatrange], ax=axes[1]).set(
                    title="Max"
                )
            elif plottype == "line":
                sns.lineplot(data=[min for min, _ in normdatrange], ax=axes[0]).set(
                    title="Min"
                )
                sns.lineplot(data=[max for _, max in normdatrange], ax=axes[1]).set(
                    title="Max"
                )
        else:
            normdatrange = [x for x in self.get_normalization_data_range(task) if x]
            graph_titles, graph_data = self.flatten_dict(normdatrange)
            _, axes = plt.subplots(
                len(graph_titles), 2, figsize=(12, 10), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot([min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.histplot([max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot([min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.distplot([max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(data=[min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.lineplot(data=[max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1

    ##################################################
    # Backward Transfer Ratio
    ##################################################

    def get_backward_transfer_ratio_helper(
        self, df: pd.DataFrame, task_a: str = "", task_b: str = ""
    ) -> Union[None, int, dict, list]:
        try:
            if not task_a:
                return df.root.backward_transfer_ratio.iloc[0, 0]
            elif not task_b:
                return self.df2dict(
                    df.root.task_metrics[task_a].backward_transfer_ratio
                )
            else:
                return (
                    df.root.task_metrics[task_a]
                    .backward_transfer_ratio[task_b]
                    .tolist()
                )
        except (KeyError, AttributeError):
            pass

    def get_backward_transfer_ratio(
        self, task_a: str = "", task_b: str = "", run_id: str = ""
    ) -> List[Union[None, int, dict, list]]:
        if not run_id:
            return [
                self.get_backward_transfer_ratio_helper(run, task_a, task_b)
                for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_backward_transfer_ratio_helper(run, task_a, task_b)

    def plot_backward_transfer_ratio(
        self, plottype: str, task_a: str = "", task_b: str = ""
    ):
        if not task_a:
            graph_data = [x for x in self.get_backward_transfer_ratio() if x]
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)
        elif not task_b:
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_backward_transfer_ratio(task_a) if x]
            )
            # print(graph_titles,graph_data)
            _, axes = plt.subplots(
                len(graph_titles), 1, figsize=(12, 5), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(data=list(v), ax=axes[i]).set(title=k)
                    i += 1
        else:
            graph_data = self.flatten_list(
                [x for x in self.get_backward_transfer_ratio(task_a, task_b) if x]
            )
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Forward Transfer Ratio
    ##################################################

    def get_forward_transfer_ratio_helper(
        self, df: pd.DataFrame, task_a: str = "", task_b: str = ""
    ) -> Union[None, int, dict, list]:
        try:
            if not task_a:
                return df.root.forward_transfer_ratio.iloc[0, 0]
            elif not task_b:
                return self.df2dict(df.root.task_metrics[task_a].forward_transfer_ratio)
            else:
                return (
                    df.root.task_metrics[task_a]
                    .forward_transfer_ratio[task_b]
                    .tolist()[0]
                )
        except (KeyError, AttributeError):
            pass

    def get_forward_transfer_ratio(
        self, task_a: str = "", task_b: str = "", run_id: str = ""
    ) -> List[Union[None, int, dict, list]]:
        if not run_id:
            return [
                self.get_forward_transfer_ratio_helper(run, task_a, task_b)
                for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_forward_transfer_ratio_helper(run, task_a, task_b)

    def plot_forward_transfer_ratio(
        self, plottype: str, task_a: str = "", task_b: str = ""
    ):
        if not task_a:
            graph_data = [x for x in self.get_forward_transfer_ratio() if x]
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)
        elif not task_b:
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_forward_transfer_ratio(task_a) if x]
            )
            _, axes = plt.subplots(
                len(graph_titles), 1, figsize=(12, 5), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
        else:
            graph_data = self.flatten_list(
                [x for x in self.get_forward_transfer_ratio(task_a, task_b) if x]
            )
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Backward Transfer Contrast
    ##################################################

    def get_backward_transfer_contrast_helper(
        self, df: pd.DataFrame, task_a: str = "", task_b: str = ""
    ) -> Union[None, int, dict, list]:
        try:
            if not task_a:
                return df.root.backward_transfer_contrast.iloc[0, 0]
            elif not task_b:
                return self.df2dict(
                    df.root.task_metrics[task_a].backward_transfer_contrast
                )
            else:
                return (
                    df.root.task_metrics[task_a]
                    .backward_transfer_contrast[task_b]
                    .tolist()[0]
                )
        except (KeyError, AttributeError):
            pass

    def get_backward_transfer_contrast(
        self, task_a: str = "", task_b: str = "", run_id: str = ""
    ) -> List[Union[None, int, dict, list]]:
        if not run_id:
            return [
                self.get_backward_transfer_contrast_helper(run, task_a, task_b)
                for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_backward_transfer_contrast_helper(
                        run, task_a, task_b
                    )

    def plot_backward_transfer_contrast(
        self, plottype: str, task_a: str = "", task_b: str = ""
    ):
        if not task_a:
            graph_data = [x for x in self.get_backward_transfer_contrast() if x]
            fig, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)
        elif not task_b:
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_backward_transfer_contrast(task_a) if x]
            )
            # print(graph_titles,graph_data)
            _, axes = plt.subplots(
                len(graph_titles), 1, figsize=(12, 5), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
        else:
            graph_data = self.flatten_list(
                [x for x in self.get_backward_transfer_contrast(task_a, task_b) if x]
            )
            fig, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Forward Transfer Contrast
    ##################################################

    def get_forward_transfer_contrast_helper(
        self, df: pd.DataFrame, task_a: str = "", task_b: str = ""
    ) -> Union[None, int, dict, list]:
        try:
            if not task_a:
                return df.root.forward_transfer_contrast.iloc[0, 0]
            elif not task_b:
                return self.df2dict(
                    df.root.task_metrics[task_a].forward_transfer_contrast
                )
            else:
                return (
                    df.root.task_metrics[task_a]
                    .forward_transfer_contrast[task_b]
                    .tolist()[0]
                )
        except (KeyError, AttributeError):
            pass

    def get_forward_transfer_contrast(
        self, task_a: str = "", task_b: str = "", run_id: str = ""
    ) -> List[Union[None, int, dict, list]]:
        if not run_id:
            return [
                self.get_forward_transfer_contrast_helper(run, task_a, task_b)
                for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_forward_transfer_contrast_helper(
                        run, task_a, task_b
                    )

    def plot_forward_transfer_contrast(
        self, plottype: str, task_a: str = "", task_b: str = ""
    ):
        if not task_a:
            graph_data = [x for x in self.get_forward_transfer_contrast() if x]
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)
        elif not task_b:
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_forward_transfer_contrast(task_a) if x]
            )
            # print(graph_titles,graph_data)
            _, axes = plt.subplots(
                len(graph_titles), 1, figsize=(12, 5), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(list(v), ax=axes[i]).set(title=k)
                    i += 1
        else:
            graph_data = self.flatten_list(
                [x for x in self.get_forward_transfer_contrast(task_a, task_b) if x]
            )
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Maintenance Values MRLEP
    ##################################################

    def _get_maintenance_val_mrlep_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, list]:
        try:
            if not task:
                return [
                    df.root.task_metrics[t].maintenance_val_mrlep.iloc[0, 0]
                    for t in self.get_json_task_names()
                ]
            else:
                return df.root.task_metrics[task].maintenance_val_mrlep.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_maintenance_val_mrlep(self, task: str, run_id: str = "") -> List[list]:
        if not run_id:
            return list(
                filter(
                    None,
                    [
                        self._get_maintenance_val_mrlep_helper(run, task)
                        for run in self.dfs
                    ],
                )
            )
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return list(self._get_maintenance_val_mrlep_helper(run, task))

    def plot_maintenance_val_mrlep(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_maintenance_val_mrlep(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Maintenance Values MRTLP
    ##################################################

    def _get_maintenance_val_mrtlp_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, list]:
        try:
            if not task:
                return [
                    df.root.task_metrics[t].maintenance_val_mrtlp.iloc[0, 0]
                    for t in self.get_json_task_names()
                ]
            else:
                return df.root.task_metrics[task].maintenance_val_mrtlp.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_maintenance_val_mrtlp(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [
                self._get_maintenance_val_mrtlp_helper(run, task) for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_maintenance_val_mrtlp_helper(run, task)

    def plot_maintenance_val_mrtlp(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_maintenance_val_mrtlp(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Recovery Times
    ##################################################

    def _get_recovery_times_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, list]:
        try:
            if not task:
                return [
                    df.root.task_metrics[t].recovery_times.iloc[0, 0]
                    for t in self.get_json_task_names()
                ]
            else:
                return df.root.task_metrics[task].recovery_times.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_recovery_times(self, task: str = "", run_id: str = "") -> List[list]:
        if not run_id:
            return self.flatten_list(
                [self._get_recovery_times_helper(run, task) for run in self.dfs]
            )
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_recovery_times_helper(run, task)

    def plot_recovery_times(self, plottype: str, task: str = "") -> None:
        graph_data = self.flatten_list([x for x in self.get_recovery_times(task) if x])
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Performance Recovery
    ##################################################

    def _get_perf_recovery_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, int]:
        try:
            if not task:
                return df.root.perf_recovery.iloc[0, 0]
            else:
                return df.root.task_metrics[task].perf_recovery.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_perf_recovery(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, int]]:
        if not run_id:
            return [self._get_perf_recovery_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_perf_recovery_helper(run, task)

    def plot_perf_recovery(self, plottype: str, task: str = "") -> None:
        graph_data = [x for x in self.get_perf_recovery(task) if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Performance Maintenance MRLEP
    ##################################################

    def get_perf_maintenance_mrlep_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, int]:
        try:
            if not task:
                return df.root.perf_maintenance_mrlep.iloc[0, 0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrlep.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_perf_maintenance_mrlep(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, int]]:
        if not run_id:
            return [
                self.get_perf_maintenance_mrlep_helper(run, task) for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_perf_maintenance_mrlep_helper(run, task)

    def plot_perf_maintenance_mrlep(self, plottype: str, task: str = "") -> None:
        graph_data = [x for x in self.get_perf_maintenance_mrlep(task) if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Performance Maintenance MRTLP
    ##################################################

    def get_perf_maintenance_mrtlp_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, int]:
        try:
            if not task:
                return df.root.perf_maintenance_mrtlp.iloc[0, 0]
            else:
                return df.root.task_metrics[task].perf_maintenance_mrtlp.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_perf_maintenance_mrtlp(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, int]]:
        if not run_id:
            return [
                self.get_perf_maintenance_mrtlp_helper(run, task) for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self.get_perf_maintenance_mrtlp_helper(run, task)

    def plot_perf_maintenance_mrtlp(self, plottype: str, task: str = "") -> None:
        graph_data = [x for x in self.get_perf_maintenance_mrtlp(task) if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Relative Performance
    ##################################################

    def _get_ste_rel_perf_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, int]:
        try:
            if not task:
                return df.root.ste_rel_perf.iloc[0, 0]
            else:
                return df.root.task_metrics[task].ste_rel_perf.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_ste_rel_perf(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, int]]:
        if not run_id:
            return [self._get_ste_rel_perf_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_ste_rel_perf_helper(run, task)

    def plot_ste_rel_perf(self, plottype: str, task: str = "") -> None:
        graph_data = [x for x in self.get_ste_rel_perf(task) if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency
    ##################################################

    def _get_sample_efficiency_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, dict, int]:
        try:
            if not task:
                return df.root.sample_efficiency.iloc[0, 0]
            elif task == "all":
                # print({t:df.root.task_metrics[t].sample_efficiency.iloc[0,0] for t in df.root.task_metrics.columns.levels[0]})
                return {
                    t: df.root.task_metrics[t].sample_efficiency.iloc[0, 0]
                    for t in df.root.task_metrics.columns.levels[0]
                }
            else:
                return df.root.task_metrics[task].sample_efficiency.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_sample_efficiency(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, dict, int]]:
        if not run_id:
            return [self._get_sample_efficiency_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_sample_efficiency_helper(run, task)

    def plot_sample_efficiency(self, plottype: str, task: str = "") -> None:
        if task == "all":
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_sample_efficiency(task)]
            )
            _, axes = plt.subplots(
                len(graph_titles), 1, figsize=(12, 10), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot([x for x in v], ax=axes[i]).set(
                        title=k + " Sample Efficiency"
                    )
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot([x for x in v], ax=axes[i]).set(
                        title=k + " Sample Efficiency"
                    )
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(data=[x for x in v], ax=axes[i]).set(
                        title=k + " Sample Efficiency"
                    )
                    i += 1
        else:
            graph_data = [x for x in self.get_sample_efficiency(task) if x]
            _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([x for x in graph_data], ax=axes)
            elif plottype == "dist":
                sns.distplot([x for x in graph_data], ax=axes)
            elif plottype == "line":
                sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Run ID
    ##################################################

    def _get_run_id_helper(self, df: pd.DataFrame) -> str:
        return df.root.run_id.iloc[0, 0]

    def get_run_id(self) -> List[str]:
        return [self._get_run_id_helper(run) for run in self.dfs]

    ##################################################
    # Complexity
    ##################################################

    def _get_complexity_helper(self, df: pd.DataFrame) -> str:
        return df.root.complexity.iloc[0, 0]

    def get_complexity(self) -> List[str]:
        return [self._get_complexity_helper(run) for run in self.dfs]

    ##################################################
    # Difficulty
    ##################################################

    def _get_difficulty_helper(self, df: pd.DataFrame) -> str:
        return df.root.difficulty.iloc[0, 0]

    def get_difficulty(
        self,
    ) -> List[str]:
        return [self._get_difficulty_helper(run) for run in self.dfs]

    ##################################################
    # Scenario Type
    ##################################################

    def _get_scenario_type_helper(self, df: pd.DataFrame) -> str:
        return df.root.scenario_type.iloc[0, 0]

    def get_scenario_type(self) -> List[str]:
        return [self._get_scenario_type_helper(run) for run in self.dfs]

    ##################################################
    # Performance Measure
    ##################################################

    def _get_perf_measure_helper(self, df: pd.DataFrame) -> str:
        return df.root.metrics_column.iloc[0, 0]

    def get_perf_measure(self) -> List[str]:
        return [self._get_perf_measure_helper(run) for run in self.dfs]

    ##################################################
    # Min and Max
    ##################################################

    def _get_min_max_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, Tuple[int, int], dict]:
        try:
            if not task:
                return df.root["min"].iloc[0, 0], df.root["max"].iloc[0, 0]
            elif task == "all":
                return {
                    t: (
                        df.root.task_metrics[t]["min"].iloc[0, 0],
                        df.root.task_metrics[t]["max"].iloc[0, 0],
                    )
                    for t in df.root.task_metrics.columns.levels[0]
                }
            else:
                return (
                    df.root.task_metrics[task]["min"].iloc[0, 0],
                    df.root.task_metrics[task]["max"].iloc[0, 0],
                )
        except (KeyError, AttributeError):
            pass

    def get_min_max(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, Tuple[int, int], dict]]:
        if not run_id:
            return [self._get_min_max_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_min_max_helper(run, task)

    def plot_min_max(self, plottype: str, task: str = "") -> None:
        if task == "all":
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_min_max(task) if x]
            )
            _, axes = plt.subplots(
                len(graph_titles), 2, figsize=(12, 10), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot([min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.histplot([max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot([min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.distplot([max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(data=[min for min, _ in v], ax=axes[i][0]).set(
                        title=k + " Min"
                    )
                    sns.lineplot(data=[max for _, max in v], ax=axes[i][1]).set(
                        title=k + " Max"
                    )
                    i += 1
        else:
            graph_data = [x for x in self.get_min_max(task) if x]
            _, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([min for min, _ in graph_data], ax=axes[0]).set(
                    title="Min"
                )
                sns.histplot([max for _, max in graph_data], ax=axes[1]).set(
                    title="Max"
                )
            elif plottype == "dist":
                sns.distplot([min for min, _ in graph_data], ax=axes[0]).set(
                    title="Min"
                )
                sns.distplot([max for _, max in graph_data], ax=axes[1]).set(
                    title="Max"
                )
            elif plottype == "line":
                sns.lineplot(data=[min for min, _ in graph_data], ax=axes[0]).set(
                    title="Min"
                )
                sns.lineplot(data=[max for _, max in graph_data], ax=axes[1]).set(
                    title="Max"
                )

    ##################################################
    # Number of Experiences
    ##################################################

    def _get_num_lx_ex_helper(
        self, df: pd.DataFrame, task: str = ""
    ) -> Union[None, Tuple[int, int], dict]:
        try:
            if not task:
                return df.root.num_lx.iloc[0, 0], df.root.num_ex.iloc[0, 0]
            elif task == "all":
                return {
                    t: (
                        df.root.task_metrics[t].num_lx.iloc[0, 0],
                        df.root.task_metrics[t].num_ex.iloc[0, 0],
                    )
                    for t in df.root.task_metrics.columns.levels[0]
                }
            else:
                return (
                    df.root.task_metrics[task].num_lx.iloc[0, 0],
                    df.root.task_metrics[task].num_ex.iloc[0, 0],
                )
        except (KeyError, AttributeError):
            pass

    def get_num_lx_ex(
        self, task: str = "", run_id: str = ""
    ) -> List[Union[None, Tuple[int, int], dict]]:
        if not run_id:
            return [self._get_num_lx_ex_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_num_lx_ex_helper(run, task)

    def plot_num_lx_ex(self, plottype: str, task: str = "") -> None:
        if task == "all":
            graph_titles, graph_data = self.flatten_dict(
                [x for x in self.get_num_lx_ex(task) if x]
            )
            _, axes = plt.subplots(
                len(graph_titles), 2, figsize=(12, 10), constrained_layout=True
            )
            if plottype == "hist":
                i = 0
                for k, v in graph_data.items():
                    sns.histplot([num_lx for num_lx, _ in v], ax=axes[i][0]).set(
                        title=k + " Num_lx"
                    )
                    sns.histplot([num_ex for _, num_ex in v], ax=axes[i][1]).set(
                        title=k + " Num_ex"
                    )
                    i += 1
            elif plottype == "dist":
                i = 0
                for k, v in graph_data.items():
                    sns.distplot([num_lx for num_lx, _ in v], ax=axes[i][0]).set(
                        title=k + " Num_lx"
                    )
                    sns.distplot([num_ex for _, num_ex in v], ax=axes[i][1]).set(
                        title=k + " Num_ex"
                    )
                    i += 1
            elif plottype == "line":
                i = 0
                for k, v in graph_data.items():
                    sns.lineplot(data=[num_lx for num_lx, _ in v], ax=axes[i][0]).set(
                        title=k + " Num_lx"
                    )
                    sns.lineplot(data=[num_ex for _, num_ex in v], ax=axes[i][1]).set(
                        title=k + " Num_ex"
                    )
                    i += 1
        else:
            graph_data = [x for x in self.get_num_lx_ex(task) if x]
            _, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            if plottype == "hist":
                sns.histplot([num_lx for num_lx, _ in graph_data], ax=axes[0]).set(
                    title="Num_lx"
                )
                sns.histplot([num_ex for _, num_ex in graph_data], ax=axes[1]).set(
                    title="Num_ex"
                )
            elif plottype == "dist":
                sns.distplot([num_lx for num_lx, _ in graph_data], ax=axes[0]).set(
                    title="Num_lx"
                )
                sns.distplot([num_ex for _, num_ex in graph_data], ax=axes[1]).set(
                    title="Num_ex"
                )
            elif plottype == "line":
                sns.lineplot(data=[num_lx for num_lx, _ in graph_data], ax=axes[0]).set(
                    title="Num_lx"
                )
                sns.lineplot(data=[num_ex for _, num_ex in graph_data], ax=axes[1]).set(
                    title="Num_ex"
                )

    ##################################################
    # Relative Performance Values
    ##################################################

    def _get_ste_rel_perf_vals_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].ste_rel_perf_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_ste_rel_perf_vals(
        self, task: str, run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [self._get_ste_rel_perf_vals_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_ste_rel_perf_vals_helper(run, task)

    def plot_ste_rel_perf_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_ste_rel_perf_vals(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # STE Saturation Values
    ##################################################

    def _get_ste_sat_vals_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].ste_saturation_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_ste_sat_vals(self, task: str, run_id: str = "") -> List[Union[None, list]]:
        if not run_id:
            return [self._get_ste_sat_vals_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_ste_sat_vals_helper(run, task)

    def plot_ste_sat_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list([x for x in self.get_ste_sat_vals(task) if x])
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # STE Experience to Saturation Values
    ##################################################

    def _get_ste_exp_to_sat_vals_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].ste_exp_to_sat_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_ste_exp_to_sat_vals(
        self, task: str, run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [self._get_ste_exp_to_sat_vals_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_ste_exp_to_sat_vals_helper(run, task)

    def plot_ste_exp_to_sat_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_ste_exp_to_sat_vals(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency Saturation Values
    ##################################################

    def _get_se_sat_vals_helper(self, df: pd.DataFrame, task: str) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].se_saturation_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_se_sat_vals(self, task: str, run_id: str = "") -> List[Union[None, list]]:
        if not run_id:
            return [self._get_se_sat_vals_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_se_sat_vals_helper(run, task)

    def plot_se_sat_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list([x for x in self.get_se_sat_vals(task) if x])
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency Experience to Saturation Values
    ##################################################

    def _get_se_exp_to_sat_vals_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].se_exp_to_sat_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_se_exp_to_sat_vals(
        self, task: str, run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [self._get_se_exp_to_sat_vals_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_se_exp_to_sat_vals_helper(run, task)

    def plot_se_exp_to_sat_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_se_exp_to_sat_vals(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency Values
    ##################################################

    def _get_sample_efficiency_vals_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].sample_efficiency_vals.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_sample_efficiency_vals(
        self, task: str, run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [
                self._get_sample_efficiency_vals_helper(run, task) for run in self.dfs
            ]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_sample_efficiency_vals_helper(run, task)

    def plot_sample_efficiency_vals(self, plottype: str, task: str) -> None:
        graph_data = self.flatten_list(
            [x for x in self.get_sample_efficiency_vals(task) if x]
        )
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency Task Saturation Values
    ##################################################

    def _get_se_task_sat_helper(self, df: pd.DataFrame, task: str) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].se_task_saturation.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_se_task_sat(self, task: str, run_id: str = "") -> List[Union[None, list]]:
        if not run_id:
            return [self._get_se_task_sat_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_se_task_sat_helper(run, task)

    def plot_se_task_sat(self, plottype: str, task: str) -> None:
        graph_data = [x for x in self.get_se_task_sat(task) if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Sample Efficiency Task Experience to Saturation Values
    ##################################################

    def _get_se_task_exp_to_sat_helper(
        self, df: pd.DataFrame, task: str
    ) -> Union[None, list]:
        try:
            return df.root.task_metrics[task].se_task_exp_to_sat.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_se_task_exp_to_sat(
        self, task: str, run_id: str = ""
    ) -> List[Union[None, list]]:
        if not run_id:
            return [self._get_se_task_exp_to_sat_helper(run, task) for run in self.dfs]
        else:
            for run in self.dfs:
                if run_id == run.root.run_id.iloc[0, 0]:
                    return self._get_se_task_exp_to_sat_helper(run, task)

    def plot_se_task_exp_to_sat(self, plottype: str, task: str) -> None:
        graph_data = [x for x in self.get_se_task_exp_to_sat(task) if x]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=ax)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=ax)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=ax)

    ##################################################
    # Runtime
    ##################################################

    def _get_runtime_helper(self, df: pd.DataFrame) -> Union[None, int]:
        try:
            return df.root.runtime.iloc[0, 0]
        except (KeyError, AttributeError):
            pass

    def get_runtime(self) -> List[Union[None, int]]:
        return [self._get_runtime_helper(run) for run in self.dfs]

    def plot_runtime(self, plottype: str) -> None:
        graph_data = [x for x in self.get_runtime() if x]
        _, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        if plottype == "hist":
            sns.histplot([x for x in graph_data], ax=axes)
        elif plottype == "dist":
            sns.distplot([x for x in graph_data], ax=axes)
        elif plottype == "line":
            sns.lineplot(data=[x for x in graph_data], ax=axes)

    ##################################################
    # Task Names
    ##################################################

    def _get_json_task_names_helper(self, df: pd.DataFrame) -> Union[None, list]:
        try:
            return df.root.task_metrics.columns.levels[0]
        except (KeyError, AttributeError):
            pass

    def get_json_task_names(self) -> list:
        return list(
            set(
                self.flatten_list(
                    [list(self._get_json_task_names_helper(run)) for run in self.dfs]
                )
            )
        )

    ##################################################
    # L2Metrics Distribution Plot
    ##################################################

    def plot_l2metrics_dist(self) -> None:
        _, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

        vals = self.get_perf_recovery()
        sns.distplot(vals, ax=axes[0, 0])
        mean_val = np.nanmean(vals)
        axes[0, 0].axvline(x=mean_val, color="r", linestyle="--")
        axes[0, 0].set(title=f"Performance Recovery, σ = {mean_val:.2f}")

        vals = self.get_perf_maintenance_mrlep()
        sns.distplot(vals, ax=axes[0, 1])
        mean_val = np.nanmean(vals)
        axes[0, 1].axvline(x=mean_val, color="r", linestyle="--")
        axes[0, 1].set(title=f"Performance Maintenance, σ = {mean_val:.2f}")

        vals = self.get_forward_transfer_ratio()
        sns.distplot(vals, ax=axes[0, 2])
        mean_val = np.nanmean(vals)
        axes[0, 2].axvline(x=mean_val, color="r", linestyle="--")
        axes[0, 2].set(title=f"Forward Transfer, σ = {mean_val:.2f}")

        vals = self.get_backward_transfer_ratio()
        sns.distplot(vals, ax=axes[1, 0])
        mean_val = np.nanmean(vals)
        axes[1, 0].axvline(x=mean_val, color="r", linestyle="--")
        axes[1, 0].set(title=f"Backward Transfer, σ = {mean_val:.2f}")

        vals = self.get_ste_rel_perf()
        sns.distplot(vals, ax=axes[1, 1])
        mean_val = np.nanmean(vals)
        axes[1, 1].axvline(x=mean_val, color="r", linestyle="--")
        axes[1, 1].set(title=f"Relative Performance, σ = {mean_val:.2f}")

        vals = self.get_sample_efficiency()
        sns.distplot(vals, ax=axes[1, 2])
        mean_val = np.nanmean(vals)
        axes[1, 2].axvline(x=mean_val, color="r", linestyle="--")
        axes[1, 2].set(title=f"Sample Efficiency, σ = {mean_val:.2f}")

    ##################################################
    # TSV methods
    ##################################################

    def get_regime(self):
        return self.df_tsv.regime_num

    def get_regime_by_task(self, task: str):
        return self.df_tsv[self.df_tsv.task_name == task].regime_num

    def get_regime_by_block_type(self, block_type: str = "", subtype: str = ""):
        if subtype and block_type:
            return self.df_tsv[
                (self.df_tsv.block_type == block_type)
                & (self.df_tsv.block_subtype == subtype)
            ].regime_num
        elif subtype:
            return self.df_tsv[self.df_tsv.block_subtype == subtype].regime_num
        else:
            return self.df_tsv[self.df_tsv.block_type == block_type].regime_num

    def get_term_perf(self, run_id: str):
        return self.df_tsv[self.df_tsv.run_id == run_id].term_perf

    def get_tsv_task_names(self):
        return list(self.df_tsv.task_name.unique())
