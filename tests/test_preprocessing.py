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
import numpy as np
import pandas as pd
from l2metrics._localutil import smooth
from l2metrics.normalizer import Normalizer


def test_smoothing():
    x = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    assert np.allclose(
        smooth(x, window_len=3),
        (2, 1.66666667, 2, 3, 4, 5, 6, 7, 8, 9),
        rtol=1e-05,
        atol=1e-08,
    )


def test_clamp_outliers_quantiles():
    x = np.linspace(0, 100, 100)
    lower_bound, upper_bound = np.quantile(x, (0.1, 0.9))
    clamped_x = x.clip(lower_bound, upper_bound)
    assert (lower_bound, upper_bound) == (clamped_x.min(), clamped_x.max())


def test_clamp_outliers_range():
    x = np.linspace(0, 100, 100)
    lower_bound = 10
    upper_bound = 90
    clamped_x = x.clip(lower_bound, upper_bound)
    assert (lower_bound, upper_bound) == (clamped_x.min(), clamped_x.max())


def test_normalization_data_range():
    # Initialise data of lists
    data = {
        "task_name": ["t1", "t1", "t1", "t2", "t2", "t2"],
        "performance": [10, 15, 30, 40, 45, 50],
    }
    df = pd.DataFrame(data)

    normalizer = Normalizer(
        perf_measure="performance",
        data=df[["task_name", "performance"]].set_index("task_name"),
    )

    assert (normalizer.run_min, normalizer.run_max, normalizer.data_range) == (
        10,
        50,
        {"t1": {"min": 10, "max": 30}, "t2": {"min": 40, "max": 50}},
    )


def test_normalization():
    # Initialise data of lists
    data = {"task_name": ["t1", "t1", "t1"], "performance": [-100, 0, 100]}
    df = pd.DataFrame(data)

    normalizer = Normalizer(
        perf_measure="performance",
        data=df[["task_name", "performance"]].set_index("task_name"),
    )

    assert (normalizer.normalize(df)["performance"].to_numpy() == (1, 51, 101)).all()
