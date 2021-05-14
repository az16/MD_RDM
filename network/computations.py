import numpy as np 
import torch
from torch import autograd 
import torch.nn as nn
import math
from scipy.sparse import linalg  as sp
from scipy import sparse as s
from statistics import geometric_mean as gm
from itertools import zip_longest
import matplotlib.pyplot as plt

def principal_eigen(p_3):
    """
    Approximates [d1..d64]^T if P3 has errors
    Approximation done using the largest eigenvector and preparing it for
    normalization.

    p_3 - comparison matrix 64x64

    returns p_3 with approximated values

    """
    result = torch.lobpcg(p_3, k=1)[1]
    result = torch.abs(result)
    for i in range(result.shape[0]):
        #print(len(result[i].shape))
        result[i] = result[i]/geometric_mean(result[i], result.shape[1], result.shape[2])
    return result.view(result.shape[0],1,8,8)

def alternating_least_squares(sparse_m, n, limit = 100, debug=False):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix

    sparse - batch of comparison matrices from ordinal layer
    n - the relative depth map id
    limit - the max amount of iterations the algorithm is supposed to run

    returns relative depth map from filled up matrix
    """
    B, H, W = sparse_m.size()
    out_size = 2**n
    filled = torch.zeros(B,1,out_size,out_size)
    #go through batch and do als for each comparison matrix
    for b in range(B):
        sparse = sparse_m[b]
        p = torch.rand((2**(2*n), 1))
        q = torch.rand((2**(2*n-2),1))
        
        rmse_record = []

        rmse_record.append(rmse(matmul(p,q.T), sparse))

        #training loop
        iteration = 0 
        while(min_eps(rmse_record) and iteration < limit):
            
            p = als_step(sparse, q)
            rmse_record.append(rmse(matmul(p,q.T), sparse))

            q = als_step(sparse.T, p)
            rmse_record.append(rmse(matmul(p,q.T), sparse))

            if debug:
                print("iteration: {:.0f}".format(iteration+1))
                print("rmse_losses: p = {0}, q = {1}".format(rmse_record[iteration], rmse_record[iteration+1]))

            iteration = iteration + 1

        #normalize with geometric mean
        p = p/quick_gm(p)
        if debug: 
            print(quick_gm(p))

        filled[b] = p.view(1,out_size,out_size)
    return filled

def min_eps(loss, eps=0.000001):
    """
    loss - list that contains all rmse losses from als method
    returns - False if delta between last two losses has become smaller than eps
              True if delta is still larger than eps
    """
    if len(loss)<2:
        return True
    else:
        #print("eps: {0}".format(np.abs(loss[-1]-loss[-2])))
        return abs(loss[-1]-loss[-2])>eps

def matmul(t1, t2):
    return torch.matmul(t1,t2)

def rmse(m1, m2):
    return torch.mean((m1-m2)**2)**0.5

def als_step(ratings, fixed_tensor, regularization_term = 0.05):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_tensor.T@fixed_tensor + torch.eye(fixed_tensor.shape[1]) * regularization_term
        b = ratings@fixed_tensor
        A_inv = torch.inverse(A)
        solve_vecs = b@A_inv
        return solve_vecs

def from_numpy(tensor):
    return torch.from_numpy(tensor)

def to_numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

def split_matrix(d_n, d_n_1):
    print("<---------Split map2pages---------->\n")
    print("Input shape for split: {0}".format((d_n.shape)))
    ratio = int(d_n.shape[2]/16)
    first = []
    second = []
    for i in range(ratio):
        for j in range(ratio):
            c_s = 16*j
            c_e = c_s + 16
            r_s = 16*i
            r_e = r_s+16
            first.append(d_n[:,:,r_s:r_e, c_s:c_e])
            second.append(d_n_1[:,:, int(r_s/2):int(r_e/2), int(c_s/2):int(c_e/2)])
    print("Depth map split into {0} pages of shape {1}.\n".format(len(first),first[0].shape))
    return first, second

def reconstruct(splits):
    #split sizes must be the same
    #first concat along 2 axis
    #then concat along 3 axis
    print("<---------Concat pages2map---------->\n")
    print("Split shape: {0}".format(splits[0].shape))
    print("Amount of pages: {0}".format(len(splits)))
    rows = []
    ratio = int(len(splits)**(1/2))
    container = None
    for i in range(ratio):
        container = splits.pop(0)
        for j in range(ratio-1):
            container = torch.cat((container, splits.pop(0)), 2)
        rows.append(container)
    
    reconstructed = torch.cat(rows, dim=3)
    
    print("Output map shape: {0}\n".format(reconstructed.shape))

    return reconstructed

def geometric_mean(iterable, r, c):
    return torch.prod(torch.pow(iterable,1/(r*c)),0)

def quick_gm(t):
    """
    computes geometric mean for als filled matrices
    """
    exp = 1/256 #hardcoded as sizes above 16x16 are not computed
    torch.squeeze(t)
    geomean = torch.prod(torch.pow(t,exp),0)
    #print("Geometric mean: {0}".format(geomean))
    return geomean[0]

def get_size(id):
    if id == 3:
        return 8,8
    elif id == 4:
        return 16,8
    elif id == 5:
        return 32,16
    elif id == 6:
        return 64,32
    elif id == 7:
        return 128,64

def get_resized_area(r_s, r_e, c_s, c_e, dn_1):
    """
    r_s - row start index
    r_e - row end index
    c_s - column start index
    c_e - column end index
    dn_1 - depth map of size n-1

    returns the 3x3 area in the previous depth map
    """
    kernel_r1 = dn_1[0][0][r_s][c_s:c_e]
    kernel_r2 = dn_1[0][0][r_s+1][c_s:c_e]
    kernel_r3 = dn_1[0][0][r_e][c_s:c_e]

    result = torch.empty(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2],dn_1.shape[3])
    result[0][0][r_s][c_s:c_e] = kernel_r1
    result[0][0][r_s+1][c_s:c_e] = kernel_r2
    result[0][0][r_e][c_s:c_e] = kernel_r3
    result = torch.reshape(result,(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2]*dn_1.shape[3]))
    # print("Result")
    # print(result)
    return result

def find_nans(container):
    """
    checks if nans are contained in a list of tensors
    returns True if it does, False if it doesn't 
    """
    for tensor in container:
       if torch.any(tensor.isnan()):
           return True
    
    return False

def resize(depth_map, newsize):
    
    depth_map = depth_map.double()
    return nn.functional.interpolate(depth_map,size=newsize)

def upsample(depth_map):
    depth_map = depth_map.double()
    m = nn.Upsample(scale_factor=2, mode='nearest')
    return m(depth_map)

def multi_upsample(depth_map, n):
    if n == 0:
        return depth_map
    elif n > 0:
        return multi_upsample(upsample(depth_map), n-1)

def decompose_depth_map(container, dn, n, relative_map=False):
    """
    container - list that holds all calculated fine detail maps Fn
    depth_map - the current depth map that is decomposed
    n - the id of the fine detail map that is supposed to be
        caluclated
    
    return - list of fine detail maps obtained by using recursive 
             hadamard devision (elementwise division)
    """
    if n == 0:
        if not relative_map:
            container.append(dn)#append d_0
        print("\nDecomposed into {0} fine detail maps.".format(len(container)))
        print("NaN values found? -> {0}".format(find_nans(container)))
        return container
    elif n >= 1:
        dn_1 = resize(dn, 2**(n-1))
        fn = dn / upsample(dn_1)
        container.append(fn)
        return decompose_depth_map(container, dn_1, n-1, relative_map)

def recombination(list_of_components, n=7):
    """
    This method combines the optimal candidates for each fine detail map. Therefore 
    output received from the depth estimation net can't be fed directly to this method.
    In a previous step the optimal components have to be chosen.

    list_of_components - list of optimal recombination candidates for 
                         one input image (sorted after id in ascending order)
    n - the id of the depth map that is supposed to be recombined (n=7 by default since we want a 128x128 map)
    returns - Reconstructed depthmap in log scale according to formula (6) from paper
    """
    d_0 = torch.log(multi_upsample(list_of_components.pop(0), n))

    result =  torch.log(multi_upsample(list_of_components.pop(0), n-1))
    for i in range(n-2):
        #print(len(list_of_components), i)
        result = result+torch.log(multi_upsample(list_of_components[i], n-(i+2)))
    
    optimal_map = d_0 + result 
    return optimal_map

def relative_fine_detail_matrix(fine_detail_rows):
    """
    fine_detail_row - list of fine detail lists obtained from relative depth maps
    returns - matrix of relative fine detail components
    """
    slots = [[] for x in range(7)]
    
    #put candidates of the same size together in lists
    for row in fine_detail_rows:
        for fine_detail_map in row:
            idx = idx_from_size(fine_detail_map)
            slots[idx].append(fine_detail_map)
    
    #create matrix from candidates
    fine_detail_matrices = [make_matrix(x) for x in slots]

    return fine_detail_matrices

def idx_from_size(fine_detail_map):
    B,C,H,W = fine_detail_map.size()

    if H == 1:
        return 0
    elif H == 2:
        return 1
    elif H == 4:
        return 2
    elif H == 8:
        return 3
    elif H == 16:
        return 4
    elif H == 32:
        return 5
    elif H == 64:
        return 6
    elif H == 128:
        return 7

def make_matrix(list_of_candidates):
    """
    Method that reshapes each candidate into a column vector a_i and horizontally
    stacks them to create a fine detail matrix A where each column is a fine detail map a_i.

    list_of_candidates - all fine detail components of same size in a list
    returns - matrix of all fine detail components
    """
    B,C,H,W = list_of_candidates[0].size()
    candidates = [x.view(B,1,C*H*W) for x in list_of_candidates]
    return torch.cat(candidates, dim=1)

def optimize_components_old(weights, yhat, y, lr=0.001):
    w = weights.weight_list

    pred = make_pred(w, yhat)
    loss = squared_err(pred,y)
    loss = [x.backward() for x in loss]
    with torch.no_grad():
        for i in range(len(yhat)) :
            #print(w[i].grad)
            weights.update(i, lr, w[i].grad)
    
    return pred

def optimize_components(weights, lr, yhat, y):
    w = weights.weight_list
    #debug_print_list(w)
    pred = make_pred(w, yhat)
    loss = squared_err(pred,y)

    optimizer = torch.optim.SGD(params=w,lr=lr)
    optimizer.zero_grad()
    loss = [x.backward() for x in loss]
    optimizer.step()

    return pred

def make_pred(w, A):
    for i in range(len(A)):
        B,M = A[i].shape[0], A[i].shape[2]
        tmp = torch.zeros((B, M, 1))

        for b in range(A[i].shape[0]):
           tmp[b]  = A[i][b].T.float()@w[i].float()
        A[i] = tmp.view(B,1,int(math.sqrt(M)),int(math.sqrt(M)))
    return A 

def squared_err(yhat,y):
    sqr_err_list = []
    for i in range(7):
        sqr_err_list.append(torch.sum(torch.abs(y[i]-yhat[i])**2))

    return sqr_err_list

def debug_recombination():
    """
    returns a list of fine detail components to test the recombination method
    """
    container = [torch.randint(1,10,(1,1,2**x,2**x)) for x in range(7)]

    return container

def debug_print_list(li):
    print("length: {0}".format(len(li)))
    for el in li:
        print(el)

if __name__ == "__main__":
    for dmap in debug_recombination():
        print(dmap.shape)

    print("recombinated shape: {0}".format(recombination(debug_recombination()).shape))
    print(recombination(debug_recombination()))


   
