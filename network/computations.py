import numpy as np 
import torch
from scipy.sparse import linalg  as sp
from scipy import sparse as s
from statistics import geometric_mean as gm
from itertools import zip_longest

   
def fill_sparse_R3(p_3):
    """
    Approximates [d1..d64]^T if P3 has errors
    Approximation done using the largest eigenvector and preparing it for
    normalization.

    p_3 - tensor containing comparison matrix 64x64

    returns p_3 with approximated values

    """

    np_p_3 = s.lil_matrix(p_3.cpu().detach().numpy())

    eig_val, eig_vec = sp.eigs(np_p_3, k=1, which='LM')
    eig_vec = np.real(eig_vec)
    eig_vec = eig_vec/geometric_mean(eig_vec, len(eig_vec), 1)

    # reciproc = np.reciprocal(eig_vec)
    # result = np.dot(reciproc,eig_vec.T)
    result = torch.reshape(torch.from_numpy(eig_vec), (1, 1, 8, 8))
    #print(result)
    return result

def fill_sparse_Rn(P, q, p, iterations):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix
    P_n,n-1

    P - the estimated comparison matrix (sparse)
    q - vector of size 2**2n-2
    p - vector of size 2**2n

    returns relative depth map from filled up matrix
    """
    P = P.cpu().detach().numpy()
    S = np.nonzero(P)
    S = np.vstack([S[0],S[1]]).T

    num_iter = 100
    rmse_record = torch.empty(2*num_iter+1, 2)
    rmse_record[:][1] = []

    # Initialization  
    component_row = torch.ones(P.shape[0], 1, P.shape[2])
    component_col = torch.ones(P.shape[1], 1, P.shape[2])
    intermediate_mat = torch.zeros(P.shape)

    for index_page in range(P.shape[2]):
        intermediate_mat[:][:][index_page] = component_row[:][:][index_page] * component_col[:][:][index_page]
    
    rmse_record[1][2] = torch.mean((torch.square(intermediate_mat - P)))**0.5

    # Repetitive ALS
    index_iter = 0
    while index_iter < num_iter:
        index_iter = index_iter + 1
        
        component_row = (sum(component_col * P.permute([2,1,3]), 1)).permute([2,1,3]) / (sum(component_col * component_col, 1)).permute([2,1,3])
        for index_page in range(P.shape[2]):
            intermediate_mat[:][:][index_page] = component_row[:][:][index_page] * component_col[:][:][index_page]
        
        rmse_record[2*index_iter+0][2] = torch.mean((torch.square(intermediate_mat - P)))**0.5
        
    
        component_col = sum(component_row * P, 1).permute([2,1,3])/sum(component_row * component_row, 1).permute([2,1,3])
        for index_page in range(P.shape[2]):
            intermediate_mat[:][:][index_page] = component_row[:][:][index_page] * component_col[:][:][index_page]
        
        rmse_record[2*index_iter+1][2] = torch.mean((torch.square(intermediate_mat - P)))**0.5
    

    output_mat = intermediate_mat
    p = component_row.permute([1,3,2])
    q = component_col.permute([1,3,2])

def split_matrix(d_n, d_n1, in_size, out_size):
    ratio = in_size/out_size
    d_split = []
    d_1_split = []
    r_dn = torch.reshape(d_n, (d_n.shape[0], d_n.shape[1], d_n.shape[2]/2*d_n.shape[3]/2))
    r_dn1 = torch.reshape(d_n1, (d_n1.shape[0], d_n1.shape[1], d_n1.shape[2]/2*d_n1.shape[3]/2))

    for i in range(ratio):
        split1 = torch.empty((d_n.shape[0], d_n.shape[1], out_size[0]))
        split2 = torch.empty((d_n1.shape[0], d_n1.shape[1], out_size[1]))

        for j in range(out_size[0]):
            index = i*out_size[0]+j
            split1[0][0] = r_dn[0][0][index]
            if j < 8:
                split2[0][0] = r_dn1[0][0][index]

        d_split.append(split1)
        d_1_split.append(split2)    

    return (d_split, d_1_split)

def geometric_mean(iterable, r, c):
    return np.array([x**(1/(r*c)) for x in iterable]).prod()

def get_size(id):
    if id == 3:
        return 8,8
    elif id == 4:
        return 8,16
    elif id == 5:
        return 16,32
    elif id == 6:
        return 32,64
    elif id == 7:
        return 64,128

def merge_into_row(size, data_to_merge, start_index):
    first_split = np.empty(start_index)
    second_split = np.empty(size-(len(first_split)+len(data_to_merge)))
    data_to_merge = data_to_merge.cpu().detach().numpy()
    result = np.hstack((first_split, data_to_merge, second_split))
    return torch.from_numpy(result)
    
def get_resized_area(r_s, r_e, c_s, c_e, dn_1):
    """
    r_s - row start index
    r_e - row end index
    c_s - column start index
    c_e - column end index
    dn_1 - depth map of size n-1

    returns the 3x3 area in the previous depth map
    """
    kernel = torch.cat((dn_1[0][0][r_s][c_s:c_e], dn_1[0][0][r_s+1][c_s:c_e], dn_1[0][0][r_e][c_s:c_e]), 0)
    #print((r_s,c_s))
    kernel = merge_into_row(dn_1.shape[2]*dn_1.shape[3], kernel, r_s*c_s)
    result = torch.empty(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2]*dn_1.shape[3])
    result[0][0][:] = kernel 

    return result