import pytest
import numpy as np
import pandas as pd

def group_adjust(vals, groups, weights):
	ctry_mean = sum(vals) / len(vals)
	column_name = ['originalVals']
	grp = []
	grp_mean = []
	demeaned = []
	for i in range(0, len(groups)):
		#create an aggregate list of group names and group_mean names
		name = 'group' + str(i)
		column_name += [name]
		grp += [name]
		grp_mean += [name + 'mean']
		# print(group)
		# print(grp_mean)
	# create a table from values list and groups lists 
	gr_table = [vals] + groups
	#construct a matrix from a table
	gr_matrix = np.matrix(gr_table)
	#transpose index and columns
	gr_matrix = gr_matrix.transpose()
	# construct data frame from a matrix
	df = pd.DataFrame(data = gr_matrix, columns = column_name)
	df['originalVals'] = pd.to_numeric(df['originalVals'], errors = 'ignore')
	# print(df)
	# Creating a pivot table which aggregates values by calculating the mean of the final group 
	for i in range(0, len(groups)):
		table = pd.pivot_table(data = df,index = [grp[i]], values = ['originalVals'], aggfunc = [np.mean])
		table = table.reset_index()
		table.columns = [grp[i], grp_mean[i]]
	df = df.merge(table, on = grp[i])
	print(df)

	# Iterate the rows of the data frame, and print each group_mean
	for index, row in df.iterrows():
		weighted_mean = weights[i-1] * ctry_mean + weights[i] * row[grp_mean[i]]
		# print(weighted_mean)
		demeaned += [(row['originalVals']) - weighted_mean]
	return demeaned

vals = [1, 2, 3, 8, 5]
grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
weights = [.65, .35]
print(group_adjust(vals, [grps_1, grps_2], weights))
		
		
		
		
		
		
		
	