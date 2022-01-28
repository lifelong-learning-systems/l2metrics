# Data Preprocessing

This document describes the data preprocessing strategy used to compute LL metrics. Note both the ordering of operations, and the specific values used for clamping and normalization. This strategy is intended for the case that there are multiple available performance curves for both LL and STE agents, but it still works when there is only one curve of each agent type.

1. For each Learning Block (across all available performance curves), smooth the performance curves with a flat window of length L, where L is the minimum of 20% of the LB’s length (in LXs) and 100 LXs. Pad the smoothed curve so that it is the same length as the raw curve.
2. Reduce the impact of outliers by clamping values with one of the following methods:
   1. If no data range is provided, then for each Task Variant in the scenario, calculate the 10% and 90% percentiles of the distribution of that Task Variant’s application-specific metric values, over the LL agent and the STE. Clamp values below the 10% and above the 90% percentiles to the those values, respectively.
   2. If a data range is provided, clamp LL and STE data to the minimum and maximum data range values specified, respectively.
3. Normalize all data to a fixed range, according to the following formula:

   ```python
   normalized_task_data = ((task_data - task_min) / (task_max - task_min) * scale) + offset
   ```

   The recommended range is from 1 to 101 (scale = 100, offset = 1), as having a minimum performance of 0 can cause jumpstart metrics to take uninformatively large values. Note that this normalization method requires some variance in the data (i.e., task_max != task_min). If there is no variance and the performance is constant, the data will be set to 0 plus the offset.
