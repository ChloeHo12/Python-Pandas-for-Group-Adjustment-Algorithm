# Group-Adjustment-Algorithm

Your task is to write the group adjustment method below. There are some unimplemented unit_tests at the bottom which also need implementation. Your solution can be pure python, pure NumPy, pure Pandas or any combination of the three.  There are multiple ways of solving this problem, be creative, use comments to explain your code.

# Group Adjust Method
The algorithm needs to do the following:

1.) For each group-list provided, calculate the means of the values for each unique group.

   For example:
   
     vals       = [  1  ,   2  ,   3  ]
     ctry_grp   = ['USA', 'USA', 'USA']
     state_grp  = ['MA' , 'MA' ,  'CT' ]

   There is only 1 country in the ctry_grp list.  So to get the means:
   
     USA_mean == mean(vals) == 2
     ctry_means = [2, 2, 2]
     
   There are 2 states, so to get the means for each state:
   
     MA_mean == mean(vals[0], vals[1]) == 1.5
     CT_mean == mean(vals[2]) == 3
     state_means = [1.5, 1.5, 3]

2.) Using the weights, calculate a weighted average of those group means
   Continuing from our example:
   
    weights = [.35, .65]
    ctry_means  = [2  , 2  , 2]
    state_means = [1.5, 1.5, 3]
    weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]

3.) Subtract the weighted average group means from each original value
   Continuing from our example:
   
     val[0] = 1
     ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
     state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
     weighted_mean = 2*.35 + .65*1.5 = 1.675
     demeaned = 1 - 1.675 = -0.675
   
   Do this for all values in the original list.

4.) Return the demeaned values

