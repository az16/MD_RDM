import numpy as np 
import torch 
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

def alternating_least_squares(P, iterations=0):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix
    P_n,n-1

    P - the estimated comparison matrix (sparse)
    returns relative depth map from filled up matrix
    """
    # P = P.cpu().detach().numpy()
    # S = np.nonzero(P)
    # S = np.vstack([S[0],S[1]]).T
    #print(P, P.shape)
    num_iter = 100
    rmse_record = np.array([list(a) for a in zip(np.array([x for x in frange(0, num_iter+2, 0.5)]),np.zeros(2*num_iter+2))])
    
    # Initialization  
    component_row = torch.ones(P.shape[0], 1, P.shape[2])
    component_col = torch.ones(P.shape[1], 1, P.shape[2])
    intermediate_mat = torch.zeros(P.shape)
    print(component_col.T.shape)
    #TODO optimize
    for index_page in range(P.shape[2]):
        for r in range(P.shape[0]):
            for c in range(P.shape[1]):
                intermediate_mat[r][c][index_page] = component_row[c][0][index_page] * component_col.T[index_page][0][c]
    
    #print(intermediate_mat)

    rmse_record[0][1] = torch.mean((torch.square(intermediate_mat - P)))**0.5

    # Repetitive ALS
    index_iter = 0
    while index_iter < num_iter:
        index_iter = index_iter + 1
        #TODO optimize
        component_row = torch.reshape(sum(component_col * P.permute([1,0,2]), 0), (1,P.shape[0],P.shape[2])).permute([1,0,2]) / torch.reshape(sum(component_col * component_col * P.permute([1,0,2]), 0),(1,P.shape[0],P.shape[2])).permute([1,0,2])
        for index_page in range(P.shape[2]):
            for r in range(P.shape[0]):
                for c in range(P.shape[1]):
                    intermediate_mat[r][c][index_page] = component_row[c][0][index_page] * component_col.T[index_page][0][c]
        
        #print(P)
        #print(intermediate_mat)
        # print(torch.square(intermediate_mat - P))
        # print(torch.mean((torch.square(intermediate_mat - P))))
        rmse_record[2*index_iter+0][1] = torch.mean((torch.square(intermediate_mat - P)))**0.5
        
    
        component_col = torch.reshape(sum(component_row * P, 0), (1,P.shape[0],P.shape[2])).permute([1,0,2])/torch.reshape(sum(component_row * component_row * P.permute([1,0,2]), 0),(1,P.shape[0],P.shape[2])).permute([1,0,2])
        #TODO optimize
        for index_page in range(P.shape[2]):
            for r in range(P.shape[0]):
                for c in range(P.shape[1]):
                    intermediate_mat[r][c][index_page] = component_row[c][0][index_page] * component_col.T[index_page][0][c]
        
        rmse_record[2*index_iter+1][1] = torch.mean((torch.square(intermediate_mat - P)))**0.5
       
    output_mat = intermediate_mat
    p = component_row.permute([0,2,1])
    q = component_col.permute([0,2,1])
    return 

def fill_intermediate(intermediate, row, col):
    inter = torch.reshape

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
    print("start index: {0}".format(start_index))
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
    kernel_r1 = dn_1[0][0][r_s][c_s:c_e]
    kernel_r2 = dn_1[0][0][r_s+1][c_s:c_e]
    kernel_r3 = dn_1[0][0][r_e][c_s:c_e]

    result = torch.ones(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2],dn_1.shape[3])
    result[0][0][r_s][c_s:c_e] = kernel_r1
    result[0][0][r_s+1][c_s:c_e] = kernel_r2
    result[0][0][r_e][c_s:c_e] = kernel_r3
    result = torch.reshape(result,(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2]*dn_1.shape[3]))
    # print("Result")
    # print(result)
    return result

def frange(start, stop=None, step=None):

    if stop == None:
        stop = start + 0.0
        start = 0.0

    if step == None:
        step = 1.0

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield (start) # return float number
        start = start + step

def depth2label_sid(depth, K=80.0, alpha=1.0, beta=90.4414):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    # if torch.cuda.is_available():
    #     alpha = alpha.cuda()
    #     beta = beta.cuda()
    #     K = K.cuda()

    label = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    label = torch.max(label, torch.zeros(label.shape)) # prevent negative label.
    # if torch.cuda.is_available():
    #     label = label.cuda()
        
    return label.int()

def label2depth_sid(label, K=80.0, alpha=1.0, beta=89.4648, gamma=-0.9766):
    if torch.cuda.is_available():
        alpha = torch.Tensor(alpha).cuda()
        beta = torch.Tensor(beta).cuda()
        K = torch.Tensor(K).cuda()
    else:
        alpha = torch.Tensor(alpha)
        beta = torch.Tensor(beta)
        K = torch.Tensor(K)

    label = label.float()
    ti_0 = torch.exp(torch.log(alpha) + torch.log(beta/alpha)*label/K) # t(i)
    ti_1 = torch.exp(torch.log(alpha) + torch.log(beta/alpha)*(label+1)/K) # t(i+1)
    depth = (ti_0 + ti_1) / 2 - gamma # avg of t(i) & t(i+1)
    return depth.float()

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C
