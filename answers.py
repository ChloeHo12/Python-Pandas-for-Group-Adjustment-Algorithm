import numpy as np
import pandas as pd
import pytest
from datetime import datetime

def group_adjust(vals, groups, weights):
    
    # Troubleshoot possible errors beforehand 
    if len(set(map(len,groups))) != 1:
        raise ValueError('Not all groups have the same lengths')
    elif len(weights) != len(groups):
        raise ValueError('Number of weight elements is not \
                        equal to number of groups')

    '''
    If we create a dataframe directly with all groups of million rows, the 
    loading operations will be extremely time-consuming (Test_performace took
    10 mins-ish)
    Alternative approach listed below:
    a) Create a dataframe with only vals column
    b) Add (group) columns from groups one-by-one
    c) Rearrange new column names (start with grps_0->grps_1->grps_2 - > ...)
    d) Perform standard Pandas operations (group by) to get new means
    e) Add those means on separate columns
    f) Multiply each means with corresponding weight value
    g) Aggregating all new means into weighted_mean
    h) Find demeaned by subtracting weighted_mean fromvals
    i) Return demeaned in form of a list
    '''
    
    # Convert list of groups + vals into dataframe
    grps_df = pd.DataFrame({'vals': vals})
    for index, group in enumerate(groups):
        grps_name = 'grps_' + str(index) # 0 -> grps_0, 1 -> grps_1
        grps_df[grps_name] = group
        
    # Create a list for new column names
    new_col = []
    
    # Find mean of each groups
    for i,col in enumerate(grps_df.iloc[:,-len(groups):].columns):
        
        # Get name of the new mean column(s) to easily address it based on orders
        mean_col = 'mean_'+str(col) #mean_grps_0, mean_grps_1 
        new_col.append(mean_col)
        
        # Calculate mean by each groups
        grps_df[mean_col] = grps_df.groupby(col).vals.transform('mean')
        
        # Multiply those groups by according scalar in weights
        grps_df[mean_col] = grps_df[mean_col] * weights[i]
    
    # Find weighted mean by totaling all previous means
    grps_df['weighted_mean'] = grps_df[new_col].sum(axis = 1)
        
    # Compute demeaned value by subtracting weighted_mean from vals
    grps_df['demeaned'] = grps_df['vals'] - grps_df['weighted_mean']
    
    # A list-like demeaned version of the input values
    return grps_df['demeaned'].tolist()

    raise NotImplementedError

def test_three_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    # 1 - (USA_mean*.15 + MA_mean * .35 + WEYMOUTH_mean * .5)
    # 2 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # 3 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # etc ...
    # Plug in the numbers ...
    # 1 - (.15 * 3.8 + .35 * 2.0 + .5 * 1.0) = -0.770
    # 2 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) = -0.520
    # 3 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) =  0.480
    # etc...

    answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5
        
def test_two_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
    # 1 - (.65 * 3.8 + .35 * 1.0) = -1.82
    # 2 - (.65 * 3.8 + .35 * 2.0) = -1.17
    # 3 - (.65 * 3.8 + .35 * 5.33333) = -1.33666
    answer = [-1.82, -1.17, -1.33666, 3.66333, 0.66333]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5
        
def test_missing_vals():
    # If you're using NumPy or Pandas, use np.NaN
    # If you're writing pyton, use None
    vals = [1, np.NaN, 3, 5, 8, 7]
    # vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)

    # This should be None or np.NaN depending on your implementation
    # please feel free to change this line to match yours
    answer = [-2.47, np.NaN, -1.170, -0.4533333, 2.54666666, 1.54666666]
    # answer = [-2.47, None, -1.170, -0.4533333, 2.54666666, 1.54666666]

    for ans, res in zip(answer, adj_vals):
        if ans is None:
            assert res is None
        elif np.isnan(ans):
            assert np.isnan(res)
        else:
            assert abs(ans - res) < 1e-5
            
def test_weights_len_equals_group_len():
    # Need to have 1 weight for each group

    # vals = [1, np.NaN, 3, 5, 8, 7]
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)
        
def test_group_len_equals_vals_len():
    # The groups need to be same shape as vals
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)

'''       
def test_performance():
    vals = 1000000*[1, None, 3, 5, 8, 7]
    # If you're doing numpy, use the np.NaN instead
    #vals = 1000000 * [1, np.NaN, 3, 5, 8, 7]
    grps_1 = 1000000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1000000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1000000 * [1, 2, 2, 3, 4, 5]
    weights = [.20, .30, .50]

    start = datetime.now()
    group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start
'''
