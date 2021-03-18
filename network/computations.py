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
    result = torch.lobpcg(p_3, k=1, largest=True)[1].cpu().detach().numpy()
    result = np.abs(result)
    for i in range(result.shape[0]):
        #print(result[i])
        result[i] = result[i]/geometric_mean(result[i], len(result[i]), 1)
    return from_numpy(result).view(result.shape[0],1,8,8)

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
        p = torch.rand((2**(2*n), 1)).double()
        q = torch.rand((2**(2*n-2),1)).double()
        
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
        return np.abs(loss[-1]-loss[-2])>eps

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
    ratio = int(np.sqrt(len(splits)))
    container = None
    for i in range(ratio):
        container = splits.pop(0)
        for j in range(ratio-1):
            container = torch.cat((container, splits.pop(0)), 2)
        rows.append(container)
    
    reconstructed = rows.pop(0)
    for entry in rows:
        reconstructed = torch.cat((reconstructed, entry), 3)
    
    print("Output map shape: {0}\n".format(reconstructed.shape))

    return reconstructed
            
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

def find_nans(container):
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
        if relative_map:
            container.append(dn)#append d_0
        print("Decomposed into {0} fine detail maps.".format(len(container)))
        print("NaN values found? --> {0}".format(find_nans(container)))
        return container
    elif n >= 1:
        dn_1 = resize(dn, 2**(n-1))
        fn = dn / upsample(dn_1)
        container.append(fn)
        return decompose_depth_map(container, dn_1, n-1)

def recombination(list_of_components, n=7):
    """
    list_of_components - list of optimal recombination candidates for 
                         one input image (sorted after id in ascending order)
    n - the id of the depth map that is supposed to be recombined
    returns - Reconstructed depthmap in log scale according to formula (6) from paper
    """
    d_0 = torch.log(multi_upsample(list_of_components.pop(0), n))

    result =  torch.log(multi_upsample(list_of_components.pop(0), n-1))
    for i in range(n-1):
        result = result+torch.log(multi_upsample(list_of_components[i], n-(i+2)))
    
    optimal_map = d_0 + result 
    return optimal_map




if __name__ == "__main__":
    t1 = torch.rand((1,1,128,128))
    result = decompose_depth_map([],t1,7)
    # t2 = torch.rand((1,1,32,32))
    # t1, t2 = split_matrix(t1, t2)
    # reconstruct(t1)
   
