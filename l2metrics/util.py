import pandas as pd

def load_logdata(log_rootdir):
    """
    Sample utility function
    """
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
         columns=['a', 'b', 'c'])
    return df
