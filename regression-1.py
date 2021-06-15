import numpy as np
import random
import csv
import math
import ast

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

# =============================================================================
def get_dataset(filename):
    dataset = [] # dataset that store info from csv file
    with open(filename, 'r', encoding='UTF-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',')
            linecount = 0
            for row in csvreader:# skip the first line
                if linecount == 0:
                    linecount += 1
                    continue
                else:
                    onerow = []
                    for i in range(1, len(row)): # skip the first element in the row
                        onerow.append(row[i])
                    dataset.append(onerow)
            
            dataset = np.array(dataset).astype(np.float)
                    
        
                        

    return dataset


def print_stats(dataset, col):
    dimension = dataset.shape #(#row, #column)
    print(dimension[0])# print the number of data point
    all_col = np.sum(dataset, axis = 0)
    col_mean = all_col[col] / dimension[0]
    print(str(round(col_mean,2)))# print the mean of the sum of the column
    col_element = dataset[: , col]
    sums = 0 # sum(xi - x)^2
    for e in col_element:
        sums = sums + (e - col_mean)**2
    young_stdev = abs(sums / (dimension[0]-1) )
    
    mature_stdev = math.sqrt(young_stdev)
    print(str(round(mature_stdev,2))) # print standard deviation
          
    pass
    

def regression(dataset, cols, betas): 
    mse = 0
    temp = betas[0]
    young_mse = []
    
    for i in range(len(dataset)):
        for j in range(len(cols)):
            temp += dataset[i][cols[j]] * betas[j + 1]
        young_mse.append(temp)
        temp = betas[0]
    for i in range(len(dataset)):
        mse += math.pow(young_mse[i] - dataset[i][0], 2) 
    mse = mse / len(dataset)
    
    return mse

 
   
def gradient_descent(dataset, cols, betas):
    grads = []
    fx = []
    temp = betas[0]
        
    for i in range(len(dataset)):
        for j in range(len(cols)):
            temp += dataset[i][cols[j]] * betas[j+1]
        fx.append(temp)
        temp = betas[0]
    
    gradient = 0
    for i in range(len(betas)):
        for j in range(len(dataset)):
            if i == 0:
                gradient += fx[j] - dataset[j][0]
            else:
                gradient += (fx[j] - dataset[j][0]) * dataset[j][cols[i-1]]
        grads.append(gradient * (2/len(dataset)))
        gradient = 0   
    return np.array(grads)

def sgd_gradient_descent(dataset, cols, betas):
    grads = []
    fx = []
    temp = betas[0]
        
    for i in range(len(dataset)):
        for j in range(len(cols)):
            temp += dataset[i][cols[j]] * betas[j+1]
        fx.append(temp)
        temp = betas[0]
    
    gradient = 0
    for i in range(len(betas)):
        for j in range(len(dataset)):
            if i == 0:
                gradient += fx[j] - dataset[j][0]
            else:
                gradient += (fx[j] - dataset[j][0]) * dataset[j][cols[i-1]]
        grads.append(gradient * 2)
        gradient = 0   
    return np.array(grads)    

def iterate_gradient(dataset, cols, betas, T, eta):
    for i in range(T):
        gd = gradient_descent(dataset, cols, betas)
        for j in range(len(betas)):
            betas[j] = betas[j] - (eta*float(gd[j]))
        print(i + 1, end=" ")
        print("%.2f"%regression(dataset, cols, betas), end = " ")
        for  j in range(len(betas)):
            print("%.2f"%betas[j], end=" ")
        print()
        

def compute_betas(dataset, cols):
    # building X
    xmatrix = []
    ymatrix = []
    for i in range(len(dataset)):
        row = [1]
        for j in cols:
            row.append(dataset[i][j])
        xmatrix.append(row)
        ymatrix.append(dataset[i][0])
    xmatrix = np.array(xmatrix)
    ymatrix = np.array(ymatrix)
    betas = list(np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xmatrix),xmatrix)),np.transpose(xmatrix)), ymatrix))
    out = []
    out.append(regression(dataset, cols, betas))
    for j in range(len(betas)):
        out.append(betas[j])
    return tuple(out)
    

def predict(dataset, cols, features):
    result = 0.0
    beta = compute_betas(dataset, cols)
    betacollec = []
    for i in range(len(beta)):
        if i == 0:
            continue
        else:
            betacollec.append(beta[i])
    
    for i in range(len(betacollec)):
        if i == 0:
            result += betacollec[i]
        else:
            result += betacollec[i]*features[i-1]
    
        
    
    
    
    return result


def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)


def sgd(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.
    You must use random_index_generator() to select individual data points.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """

    # random number generator
    var = random_index_generator(0, 252)
    for i in range(T):
        # prepare a babyset for sgd
        babyset = []
        rand_index = next(var)
        temp = dataset[rand_index]
        babyset.append(temp)
        
        
        # get the gradient for this babyset
        grd = sgd_gradient_descent(babyset, cols, betas)

        for j in range(len(betas)):
            betas[j] = betas[j] - (eta*float(grd[j]))
        print(i + 1, end=" ")
        print("%.2f"%regression(dataset, cols, betas), end = " ")
        for  j in range(len(betas)):
            print("%.2f"%betas[j], end=" ")
        print()
            
           
                

        
            
       
        
        
    pass


if __name__ == '__main__':
    
    pass
   
   