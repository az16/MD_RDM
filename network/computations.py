import numpy as np 
import torch 
import torch.nn as nn
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

    np_p_3 = s.lil_matrix(p_3.cpu().detach().numpy())

    eig_val, eig_vec = sp.eigs(np_p_3, k=1, which='LM')
    eig_vec = np.real(eig_vec)
    eig_vec = eig_vec/geometric_mean(eig_vec, len(eig_vec), 1)

    # reciproc = np.reciprocal(eig_vec)
    # result = np.dot(reciproc,eig_vec.T)
    result = (torch.from_numpy(eig_vec)).view(1, 8, 8)
    #print(result)
    return result

def als_old_version(P, iterations=0):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix
    P_n,n-1

    P - the estimated comparison matrix (sparse)
    returns relative depth map from filled up matrix
    """
    
    num_iter = iterations
    rmse_record = np.array([list(a) for a in zip(np.array([x for x in frange(0, num_iter+1, 0.5)]),np.zeros(2*num_iter+1))])
    # Initialization  
    component_row = torch.ones(P.shape[0], 1, P.shape[2]).double()
    component_col = torch.ones(P.shape[1], 1, P.shape[2]).double()
    valid = mask16x16()

    intermediate_mat = torch.zeros(P.shape)
    for index_page in range(P.shape[2]):
        intermediate_mat[:,:,index_page] = torch.matmul(component_row[:,:,index_page],component_col[:, :, index_page].T)

    rmse_record[0][1] = torch.mean((torch.square(intermediate_mat[:] - P[:] * valid)))**0.5
    #print(rmse_record)
    # Repetitive ALS
    index_iter = 0
    while index_iter < num_iter:
        index_iter = index_iter + 1
        
        #component_row = torch.unsqueeze(sum(component_col * P.permute([1,0,2]) * valid.permute([1,0,2]), 0), 1) / torch.unsqueeze(sum(component_col * component_col * valid.permute([1,0,2]), 0), 1)
        component_row = als_step(component_row.T, component_col)
       
        for index_page in range(P.shape[2]):
            intermediate_mat[:,:,index_page] = matmul(component_row[:,:,index_page], component_col[:,:,index_page].T)
        #print("intermediate = {0}".format(intermediate_mat))
        rmse_record[2*index_iter-1][1] = rmse(intermediate_mat, P)#torch.mean(torch.square((intermediate_mat - P * valid)))**0.5

        #component_col = torch.unsqueeze(sum(component_row * P * valid, 0), 1)/ torch.unsqueeze(sum(component_row * component_row * valid, 0),1)     
        component_col = als_step(component_col, component_row)
     
        for index_page in range(P.shape[2]):
                    intermediate_mat[:,:,index_page] = matmul(component_row[:,:,index_page],component_col[:,:,index_page].T)
        
        rmse_record[2*index_iter][1] = rmse(intermediate_mat, P)#torch.mean((torch.square(intermediate_mat[:] - P[:]*valid)))**0.5

        print("iteration: {:.0f}, rmse_list_len: {:.0f}".format(index_iter, 2*index_iter))
    print("max_rmse_score: {:.5f}, min_rmse_score: {:.5f}".format(max(rmse_record[:,1]), min(rmse_record[:,1])))
    output_mat = intermediate_mat
    p = component_row.permute([0,2,1])
    q = component_col.permute([0,2,1])
    print("Filled mat shape: {0}, p vector shape: {1}, q vector shape: {2}".format(output_mat.shape, p.shape, q.shape))
    return p.T

def alternating_least_squares(sparse, n, limit = 100):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix

    sparse - the estimated comparison matrix from ordinal layer
    n - the relative depth map id
    limit - the max amount of iterations the algorithm is supposed to run

    returns relative depth map from filled up matrix
    """
    sparse = sparse.view(256, 64)
    p = torch.rand((2**(2*n), 1)).double()
    q = torch.rand((2**(2*n-2),1)).double()
    
    rmse_record = []

    rmse_record.append(rmse(matmul(p,q.T), sparse))

    #training loop
    iteration = 0 
    while(iteration < limit):
        
        p = als_step(sparse, q)
        rmse_record.append(rmse(matmul(p,q.T), sparse))

        q = als_step(sparse.T, p)
        rmse_record.append(rmse(matmul(p,q.T), sparse))

        print("iteration: {:.0f}".format(iteration+1))
        print("rmse_losses: p = {0}, q = {1}".format(rmse_record[iteration], rmse_record[iteration+1]))

        iteration = iteration + 1

    p = p.view(16,16)

    return p
    
def matmul(t1, t2, numpy=False):
    if numpy:
        return np.matmul(t1,t2)
    return torch.matmul(t1,t2)

def rmse(m1, m2):
    return torch.mean((m1-m2)**2)**0.5

def als_step(ratings, fixed_tensor, regularization_term = 0.05):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        ratings = to_numpy(ratings)
        fixed_tensor = to_numpy(fixed_tensor)
        A = fixed_tensor.T.dot(fixed_tensor) + np.eye(fixed_tensor.shape[1]) * regularization_term
        b = ratings.dot(fixed_tensor)
        #print(A.shape)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        solve_vecs = from_numpy(solve_vecs)
        #print(solve_vecs.shape)
        return solve_vecs

def from_numpy(tensor):
    return torch.from_numpy(tensor)

def to_numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

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

def summarize(tensor, axis):
    result = torch.sum(tensor, axis)/tensor.shape[-1]
    return result

def cat_splits(splits):
    result = splits.pop(0)
    for split in splits:
        result = torch.cat((result, split), 2)
    
    return result

def geometric_mean(iterable, r, c):
    return np.array([x**(1/(r*c)) for x in iterable]).prod()

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

def merge_into_row(size, data_to_merge, start_index):
    print("start index: {0}".format(start_index))
    first_split = np.empty(start_index)
    second_split = np.empty(size-(len(first_split)+len(data_to_merge)))
    data_to_merge = data_to_merge.cpu().detach().numpy()
    result = np.hstack((first_split, data_to_merge, second_split))
    return torch.from_numpy(result)

def mask16x16():
    mask = torch.zeros((16,16,64))
    for index_row in range(16):
                for index_col in range(16):
                    index_resized_row = np.floor(index_row/2)
                    index_resized_col = np.floor(index_col/2)
                    index_row_start = int(min(max(index_resized_row, 0), 5))
                    index_row_end = index_row_start+2
                    index_col_start = int(min(max(index_resized_col, 0), 5))
                    index_col_end = index_col_start+3
                    comparison_area = get_resized_area(index_row_start, index_row_end, index_col_start, index_col_end, torch.ones((1,1,8,8)))
                    mask[index_row][index_col][:] = comparison_area
    return mask

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

def valid_range_maker(input_size, in_type):
    window = []
    if in_type == 1:
        window = [
            [0,1,0],
            [1,1,1],
            [0,1,0]]
    elif in_type == 2:
        window = [
            [1,1,1],
            [1,1,1],
            [1,1,1]]
    elif in_type == 4:
        window = [
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]]
    elif in_type == 5:
        window = [
            [0,1,1,1,0],
           [ 1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0]]
    elif in_type == 8:
        window = [
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1]]
    
    window = np.array(window)
    distance = np.floor(np.sqrt(in_type)).astype(int) #in range[1-2]
    
    row_size = input_size[0]**2
    col_size = input_size[1]**2
    rc_ratio = input_size[0]/input_size[1]

    valid_range = np.zeros((input_size[0], input_size[0], input_size[1], input_size[1]))

    for index_row_col in range (input_size[0]):
        for index_row_row in range (input_size[0]):
            
            index_col_col = np.ceil(index_row_col/rc_ratio) #in range [0-8]
            index_col_row = np.ceil(index_row_row/rc_ratio) #in range [0-8]
            
            index_col_col_start = (max(index_col_col-distance, 0))#[0-6]
            index_col_col_end = (min(index_col_col+distance, input_size[1])) #[1-input[1]]
            index_col_row_start = (max(index_col_row-distance, 0))#[0-6]
            index_col_row_end = (min(index_col_row+distance, input_size[1])) #[1-input[1]]
            index_window_col_start = (index_col_col_start - (index_col_col-distance))
            index_window_col_end = ((index_col_col+distance) - index_col_col_end)
            index_window_row_start = (index_col_row_start - (index_col_row-distance))
            index_window_row_end = ((index_col_row+distance) - index_col_row_end)

            idx_v1 = np.array([x for x in range(index_col_row_start,index_col_row_end.astype(int))])
            idx_v2 = np.array([x for x in range(index_col_col_start,index_col_col_end.astype(int))])
            idx_v3 = np.array([x for x in range(1+index_window_row_start.astype(int),len(window)-index_window_row_end.astype(int))])
            idx_v4 = np.array([x for x in range(1+index_window_col_start.astype(int),len(window)-index_window_col_end.astype(int))])

            print(idx_v1, idx_v2, idx_v3, idx_v4)
            if len(idx_v1 > 0):
                valid_range[index_row_row, index_row_col, idx_v1, idx_v2] = window[idx_v3, idx_v4]

    valid_range = torch.from_numpy(valid_range).view(row_size,col_size)

    return valid_range

def upsample(depth_map):
    m = nn.Upsample(scale_factor=2, mode='nearest')
    return m(depth_map)

def get_fine_detail(depth_map):
    pass
def decompose_depth_map(de):
    pass
