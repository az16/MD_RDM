import torch
from torch.functional import norm
import torch.nn as nn
import math, cmath


def principal_eigen(p_3):
    """
    Approximates [d1..d64]^T if P3 has errors
    Approximation done using the largest eigenvector and preparing it for
    normalization.

    p_3 - comparison matrix 64x64

    returns p_3 with approximated values

    """
    #print(p_3)
    r = torch.zeros((p_3.shape[0], 1, int(p_3.shape[1]**0.5), int(p_3.shape[2]**0.5)))
    #A = torch.lobpcg(p_3, k=1, method="ortho")[1]
    #A = torch.abs(A)
    for i in range(p_3.shape[0]):
        c = torch.eig(p_3[i], eigenvectors=True)
        e_vals = c[0]
        e_vecs = c[1]
        #print(e_vals)
        #print(e_vecs)
        #print(torch.view_as_complex(e_vals))
        #print(torch.abs(torch.view_as_complex(e_vals)))
        p_v = get_eigenvector_from_eigenvalue(e_vals, e_vecs)
        normed = p_v/geometric_mean(p_v, int(p_3.shape[1]**0.5), int(p_3.shape[1]**0.5))
        #print(A.shape)
        print(normed.shape)
        r[i][0] = normed.view(int(p_3.shape[1]**0.5), int(p_3.shape[1]**0.5))
    
    return r

def quadratic_als(sparse_m, cuda, n=3, limit = 30, debug = False):
    B, H, W = sparse_m.size()
    sparse_m = sparse_m.float()
    out_size = 2**n
    #go through batch and do als for each comparison matrix
    #for b in range(B):
    sparse = sparse_m
    p_s = 2**(2*n)
    #print(p_s,n)
    #q_s = 2**(2*n-2)
    p = torch.ones((B,p_s,1))
    q = torch.ones((B,p_s,1))
    if cuda:
        p=p.cuda()
        q=q.cuda()
    rmse_record = []
    vec_record = []
    rmse_record.append(rmse(matmul(p,q.view(B,1,p_s)), sparse))
    vec_record.append(p)
    #training loop
    iteration = 0 
    while(iteration < limit):
        p = als_step(sparse, q, cuda=cuda)
        rmse_record.append(rmse(matmul(p,q.view(B,1,p_s)), sparse))
        vec_record.append(p)

        q = als_step(sparse.view(B,W,H), p, cuda=cuda)
        #rmse_record.append(rmse(matmul(p,q.view(B,1,p_s)), sparse))

        if debug:
            print("iteration: {:.0f}".format(iteration+1))
            print("rmse_losses: p = {0}".format(rmse_record[iteration]))

        iteration = iteration + 1

    #choose best p approximation
    p = vec_record[rmse_record.index(min(rmse_record))]
    #normalize with geometric mean
    p = torch.div(p,quick_gm(p, H).expand(B,H).view(B,H,1))
    #print(p.shape)
    if debug: 
        print(quick_gm(p, H))

    filled = p.view(B,1,out_size,out_size)

    if cuda:
        return filled.cuda() 
    return filled 

def get_eigenvector_from_eigenvalue(e, v):
    idx = torch.topk(e, k=1, dim=0)[1][0][0]
    
    corresponding_vector = v[:, idx]

    print(corresponding_vector)
    return torch.abs(corresponding_vector)

def alternating_least_squares(sparse_m, n, cuda, limit = 30, debug=False):
    """
    Implemetation of ALS algorithm to approximate the comparison matrix

    sparse - batch of comparison matrices from ordinal layer
    n - the relative depth map id
    limit - the max amount of iterations the algorithm is supposed to run

    returns relative depth map from filled up matrix
    """
    B, H, W = sparse_m.size()
    sparse_m = sparse_m.float()
    out_size = 2**n
    filled = torch.zeros(B,1,out_size,out_size)
    #go through batch and do als for each comparison matrix
    #for b in range(B):
    sparse = sparse_m
    p_s = 2**(2*n)
    q_s = 2**(2*n-2)
    p = torch.ones((B,p_s,1))
    q = torch.ones((B,q_s,1))

    if cuda:
        p=p.cuda()
        q=q.cuda()
    
    rmse_record = []
    vec_record = []
    rmse_record.append(rmse(matmul(p,q.view(B,1,q_s)), sparse))
    vec_record.append(p)
    #training loop
    iteration = 0 
    while(iteration < limit):
        
        p = als_step(sparse, q, cuda=cuda)
        rmse_record.append(rmse(matmul(p,q.view(B,1,q_s)), sparse))
        vec_record.append(p)

        q = als_step(sparse.view(B,W,H), p, cuda=cuda)
        #rmse_record.append(rmse(matmul(p,q.view(B,1,q_s)), sparse))

        if debug:
            print("iteration: {:.0f}".format(iteration+1))
            print("rmse_losses: p = {0}".format(rmse_record[iteration]))

        iteration = iteration + 1

    #choose best p approximation
    p = vec_record[rmse_record.index(min(rmse_record))]
    #normalize with geometric mean

    p = torch.div(p,quick_gm(p,H).expand(B,H).view(B,H,1))
    #print(p.shape)
    if debug: 
        print(quick_gm(p, H).shape)

    filled = p.view(B,1,out_size,out_size)

    if cuda:
        return filled.cuda() 
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
        ## print("eps: {0}".format(np.abs(loss[-1]-loss[-2])))
        return abs(loss[-1]-loss[-2])>eps

def matmul(t1, t2):
    return torch.matmul(t1,t2)

def rmse(m1, m2):
    return torch.mean((m1-m2)**2)**0.5

def als_step(ratings, fixed_tensor, cuda, regularization_term = 0.05):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        f_b, f_h, f_w = fixed_tensor.size()
        r_b, r_h, r_w = ratings.size()
        #print(torch.eye(fixed_tensor.shape[1]))
        eye = torch.eye(f_w)    
        if cuda:
            eye=eye.cuda()
        A = matmul(fixed_tensor.view(f_b, f_w, f_h),fixed_tensor) + eye * regularization_term
        #print(A.shape)
        #print(ratings.shape, fixed_tensor.shape)
        b = ratings@fixed_tensor
        #print(b.size())
        A_inv = torch.inverse(A)
        solve_vecs = b@A_inv
        return solve_vecs

def from_numpy(tensor):
    return torch.from_numpy(tensor)

def to_numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

def split_matrix(d_n, d_n_1):
    # print("<---------Split map2pages---------->\n")
    # print("Input shape for split: {0}".format((d_n.shape)))
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
    # print("Depth map split into {0} pages of shape {1}.\n".format(len(first),first[0].shape))
    return first, second

def reconstruct(splits):
    #split sizes must be the same
    #first concat along 2 axis
    #then concat along 3 axis
    # print("<---------Concat pages2map---------->\n")
    #print("Split shape: {0}".format(splits[0].shape))
    # print("Amount of pages: {0}".format(len(splits)))
    rows = []
    ratio = int(len(splits)**(1/2))
    container = None
    for i in range(ratio):
        #container = splits.pop(0)
        #for j in range(ratio-1):
        #container = torch.cat((container, splits[0:ratio-1], 2))
        rows.append(torch.cat(splits[0:ratio], 2))
    
    reconstructed = torch.cat(rows, dim=3)
    
    # print("Output map shape: {0}\n".format(reconstructed.shape))

    return reconstructed

def geometric_mean(iterable, r, c):
    #print(torch.pow(iterable,1/(r*c)))
    return torch.prod(torch.pow(iterable,1/(r*c)),0)

def quick_gm(t,rc):
    """
    computes geometric mean for als filled matrices
    """
    rc *= rc
    exp = 1/rc #hardcoded as sizes above 16x16 are not computed
    #print(torch.pow(t,exp).shape)
    #torch.squeeze(t)
    #print(t.shape)
    geomean = torch.prod(torch.pow(t,exp),dim=1)
    ## print("Geometric mean: {0}".format(geomean))
    return geomean

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

def get_resized_area( r_s, r_e, c_s, c_e, dn_1):
    """
    r_s - row start index
    r_e - row end index
    c_s - column start index
    c_e - column end index
    dn_1 - depth map of size n-1

    returns the 3x3 area in the previous depth map
    """
    #print(dn_1)
    kernel_r1 = dn_1[:, :, r_s, c_s:c_e]
    kernel_r2 = dn_1[:, :, r_s+1, c_s:c_e]
    kernel_r3 = dn_1[:, :, r_e, c_s:c_e]
    #print(kernel_r1, kernel_r2, kernel_r3)
    result = torch.ones_like(dn_1)
    result[:, :, r_s, c_s:c_e] = kernel_r1
    result[:, :, r_s+1, c_s:c_e] = kernel_r2
    result[:, :, r_e, c_s:c_e] = kernel_r3
    #result = torch.cat((kernel_r1, kernel_r2, kernel_r3), 2)
    #print(result.shape)
    if dn_1.is_cuda:
        result = result.cuda()
    #result = torch.reshape(result,(dn_1.shape[0], dn_1.shape[1], dn_1.shape[2]*dn_1.shape[3]))
    # # print("Result")
    #print(result)
    return result.view(dn_1.shape[0], 1, dn_1.shape[2]*dn_1.shape[3])

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
    new = nn.functional.interpolate(depth_map, size=newsize)
    mask = new > 0
    #mask2 = (new <= 0) + 1
    return (new*mask)#+mask2) 

def alt_resize(depthmap, n=1):
    if n==1:
        return geometric_resize(depthmap)
    else:
        return alt_resize(geometric_resize(depthmap), n-1)

def geometric_resize(depthmap):
    """
    Reduces size of the input depthmap by computing geometric mean for
    every 4 entries in the input tensor

    depthmap - depthmap of size 2**n x 2**n
    returns - depthmap of size 2**n-1 x 2**n-1
    """
    B, C, H, W = depthmap.size()
    depthmap = depthmap.view(B,H,W)
    dn_1 = torch.zeros((B, int(H/2), int(W/2)))
    ratio = int(H/2)
    m = 2
    for i in range(ratio):
        for j in range(ratio):
            c_s = m*j
            c_e = c_s + 2
            r_s = m*i
            r_e = r_s+2
            original = depthmap[:, r_s:r_e, c_s:c_e]
            tmp = compress_entry(original)
            tmp = tmp.view(B)
            dn_1[:, i, j] = tmp

    return dn_1.view(B, 1, ratio, ratio)

def compress_entry(block):
    """
    Takes a block of 4 tensor entries and calculates the geometric mean
    returns geometric mean for block of 4 (for resizing)
    """
    B,H,W = block.size()
    result = torch.zeros((B,1))
    for b in range(B):
        result[b] = torch.prod(torch.pow(torch.flatten(block[b]),1/4),0)

    return result

def avg_resize(depthmap, n):
    #print("d_n: {0}".format(torch.isnan(depthmap).any()))
    result = depthmap
    for i in range(n):
        result = nn.AvgPool2d(kernel_size=2,stride=2)(result)
    return result

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
            ## print("D_0: {0}".format(dn))
            container.append(dn)#append d_0
        # print("\nDecomposed into {0} fine detail maps.".format(len(container)))
        # print("NaN values found? -> {0}".format(find_nans(container)))
        # for c in container:
        #     print("F_{0}: {1}".format(n, quick_gm(dn.view(dn.shape[0],dn.shape[2]*dn.shape[3],1))))
        return container
    elif n >= 1:
        #B,C,W,H = dn.size()
        #dn_1 = torch.zeros((B,C,int(W/2),int(H/2)))
        print(n)
        dn_1 = avg_resize(dn) #resize(dn, 2**(n-1))
        #print("d_n1: ({0},{1},{2})".format(torch.isnan(dn_1).any(), (dn_1<0).any(), (dn_1 == 0).any()))
        if dn.is_cuda:
            dn_1 = dn_1.cuda()

        fn = torch.div(dn,upsample(dn_1))

        # mask_n = torch.isnan(fn)
        # mask_i = torch.isinf(fn)
        # if mask_n.any():
        #     tmp = fn 
        #     fn = tmp * (mask_n == 0)
        # if mask_i.any():
        #     tmp = fn 
        #     fn = tmp * (mask_i == 0)

        #print("F_{0}: {1}".format(n, torch.abs(quick_gm(fn.view(fn.shape[0],fn.shape[2]*fn.shape[3],1)))))
        container.append(fn)
        return decompose_depth_map(container, dn_1, n-1, relative_map)

def decomp(dn, n, relative=False):
    result = []
    #print(message)
    while n > 0:
        dn_1 = avg_resize(dn, 1)
        if dn.is_cuda:
            dn_1 = dn_1.cuda()
        fn = torch.div(dn,upsample(dn_1))
        result.append(fn)
        dn = dn_1
        n -= 1

    if not relative:
        result.append(dn)

    return result

def recombination(list_of_components, n=7):
    """
    This method combines the optimal candidates for each fine detail map. Therefore 
    output received from the depth estimation net can't be fed directly to this method.
    In a previous step the optimal components have to be approximated.

    list_of_components - list of optimal recombination candidates for 
                         one input image (sorted after id in ascending order)
    n - the id of the depth map that is supposed to be recombined (n=7 by default since we want a 128x128 map)
    returns - Reconstructed depthmap in log scale according to formula (6) from paper
    """
    if list_of_components[0].shape[2] == 1:

        d_0 = multi_upsample(list_of_components.pop(0), n)

        result =  multi_upsample(list_of_components.pop(0), n-1)
        for i in range(len(list_of_components)):
            ## print(len(list_of_components), i)
            result = result+multi_upsample(list_of_components[i], n-(i+2))
        
        optimal_map = d_0 + result 
    else:
        result =  multi_upsample(list_of_components.pop(0), n-1)
        for i in range(len(list_of_components)):
            ## print(len(list_of_components), i)
            result = result+multi_upsample(list_of_components[i], n-(i+2))
        optimal_map = result
    return optimal_map

def relative_fine_detail_matrix(fine_detail_rows, cuda):
    """
    fine_detail_row - list of fine detail lists obtained from relative depth maps
    returns - matrix of relative fine detail components
    """
    slots = [[] for x in range(8)]

    #put candidates of the same size together in lists
    for row in fine_detail_rows:
        for fine_detail_map in row:
            #print(fine_detail_map.shape)
            idx = idx_from_size(fine_detail_map)
            #print(idx)
            slots[idx].append(fine_detail_map)
    
    #create matrix from candidates
    #print(fine_detail_rows)
    fine_detail_matrices = [make_matrix(x, cuda) for x in slots if not len(x) == 0]

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

def make_matrix(list_of_candidates, cuda):
    """
    Method that reshapes each candidate into a column vector a_i and horizontally
    stacks them to create a fine detail matrix A where each column is a fine detail map a_i.

    list_of_candidates - all fine detail components of same size in a list
    returns - matrix of all fine detail components
    """
    B,C,H,W = list_of_candidates[0].size()
    
    #print(cuda)
    #print(list_of_candidates)
    candidates = []
    # c_1 = []
    # c_2 = []
    if cuda:
        candidates = [torch.log(x).view(B,1,C*H*W).cuda() for x in list_of_candidates]
    else:
        # c_1 = [(x.view(B,1,C*H*W) == 0).any() for x in list_of_candidates]
        # c_2 = [(x.view(B,1,C*H*W) < 0).any() for x in list_of_candidates]
        candidates = [torch.log(x).view(B,1,C*H*W) for x in list_of_candidates]
    #t_c_nl = [torch.isnan(x).any() for x in c_nl]
    # t_c = [torch.isnan(x).any() for x in candidates]
    #print("Nan in candidates before log shift and after: before = {0}, after = {1}".format(True in t_c_nl, True in t_c))
    #print("NaN after log: {0}".format(True in t_c))
    # print("== 0, < 0, Nan = ({0},{1}, {2})".format(True in c_1, True in c_2, True in t_c))
    result = torch.cat(candidates, dim=1)
    #print(result.is_cuda)
    return result

def optimize_components_old(weights, yhat, y, lr=0.001):
    w = weights.weight_list

    pred = make_pred(w, yhat)
    loss = squared_err(pred,y)
    loss = [x.backward() for x in loss]
    with torch.no_grad():
        for i in range(len(yhat)) :
            ## print(w[i].grad)
            weights.update(i, lr, w[i].grad)
    
    return pred

def optimize_components(yhat, y, cuda):
    #debug_print_list(w)
    pred = yhat
    loss = squared_err(pred, y, cuda)
    #print(loss)
    #optimizer = torch.optim.SGD(params=w,lr=learning_rate)
    #optimizer.zero_grad()
    # loss = [x.backward() for x in loss]
    # optimizer.step()
    #print(loss)

    return pred, torch.sum(torch.as_tensor(loss)) #torch.mean(torch.as_tensor(loss))

def make_pred(w, A, cuda, relative_only):
    weights = w
    if relative_only:
        weights = w[1::]
    for i in range(len(A)):
        B, M = A[i].shape[0], A[i].shape[2]
        # print("Candidate {0}:\r".format(i))
        # print("Is nan: {0}\r".format(torch.isnan(A[i]).any()))
        if cuda:
            tmp = torch.zeros((B, M, 1)).cuda()
            for b in range(A[i].shape[0]):
                tmp[b]  = matmul(A[i][b].T.float(), weights[i].float()).cuda() 
            A[i] = tmp.view(B,1,int(math.sqrt(M)),int(math.sqrt(M))).cuda()
        else:
            tmp = torch.zeros((B, M, 1))
            for b in range(A[i].shape[0]):
                tmp[b] = matmul(A[i][b].T.float(), weights[i].float()) 
            A[i] = tmp.view(B,1,int(math.sqrt(M)),int(math.sqrt(M)))
    return A

def squared_err(yhat,y, cuda):
    sqr_err_list = []
    if yhat[0].shape[2] > y[0].shape[2]:
        y.pop(0)
    for i in range(len(yhat)):
        #if i==0:
        #print(yhat[i].is_cuda, y[i].is_cuda)
        #print("squared_err(Pred nan, Target nan) = ({0},{1})".format(torch.isnan(yhat[i]).any(), torch.isnan(y[i]).any()))
        if cuda:
            sqr_err_list.append(torch.nn.MSELoss()(yhat[i],y[i]).cuda())
        else:
            sqr_err_list.append(torch.nn.MSELoss()(yhat[i],y[i]))
        

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
        
def get_depth_sid(args, labels):
    if args == 'kitti':
        min = 0.001
        max = 80.0
        K = 71.0
    elif args == 'nyu':
        min = 0.02
        max = 10.0
        K = 68.0
    elif args == 'floorplan3d':
        min = 0.0552
        max = 10.0
        K = 68.0
    elif args == 'Structured3D':
        min = 0.02
        max = 10.0
        K = 68.0
    else:
        print('No Dataset named as ', args.dataset)

    # if torch.cuda.is_available():
    #     alpha_ = torch.tensor(min).cuda()
    #     beta_ = torch.tensor(max).cuda()
    #     K_ = torch.tensor(K).cuda()
    #else:
    alpha_ = torch.tensor(min)
    beta_ = torch.tensor(max)
    K_ = torch.tensor(K)

    # print('label size:', labels.size())
    if not alpha_ == 0.0:
        depth = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * labels / K_)
    else:
        depth = torch.exp(torch.log(beta_) * labels / K_)
    # depth = alpha_ * (beta_ / alpha_) ** (labels.float() / K_)
    # print(depth.size())
    return depth.float()

def get_labels_sid(args, depth):
    if args == 'kitti':
        alpha = 0.001
        beta = 80.0
        K = 71.0
    elif args == 'nyu':
        alpha = 0.02
        beta = 10.0
        K = 68.0
    elif args == 'floorplan3d':
        alpha = 0.0552
        beta = 10.0
        K = 68.0
    elif args == 'Structured3D':
        alpha = 0.02
        beta = 10.0
        K = 68
    else:
        print('No Dataset named as ', args.dataset)

    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    # if torch.cuda.is_available():
    #     alpha = alpha.cuda()
    #     beta = beta.cuda()
    #     K = K.cuda()
    if not alpha == 0.0:
        labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    else:
        labels = K * torch.log(depth) / torch.log(beta)
    # if torch.cuda.is_available():
    #     labels = labels.cuda()
    return labels.int()

if __name__ == "__main__":
    test = torch.abs(torch.randn((4,1,128,128)))
    #print(test)
    # r = alternating_least_squares(test,n=4, cuda=False, debug=True)
    result = decomp(test, 7, relative=True)[::-1]
    for r in result:
        print(r.shape)
    # print(r.shape)
    #print(torch. __version__ )
