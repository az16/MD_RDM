import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io
import network.computations as cp
#import computations as cp

use_cuda = True
freeze_encoder = False
class BaseModel(nn.Module):
    def load(self, path):
        # Load model from file.
        # Args:
        #    path (str): file path
        
        # parameters = torch.load(path)

        # if "optimizer" in parameters:
        #     parameters = parameters["model"]

        # self.load_state_dict(parameters)
        pass
        
        
class DepthEstimationNet(BaseModel):
    def __init__(self):
        super(DepthEstimationNet, self).__init__()

        """
        DepthEstimationNet resolutions
        Input: 226x226 3 channels
        Encoder output: 8x8 1056 channels
        Decoder outputs by id:
            * id 1, 6 => 8x8 1 channel
            * id 2, 7 => 16x16 1 channel
            * id 3, 8 => 32x32 1 channel
            * id 4, 9 => 64x64 1 channel
            * id 5, 10 => 128x128 1 channel
        """
        #GPU
        #Quantizers for Lloyd quantization
        self.quantizers = Quantization()
        #Encoder part
        self.encoder = _make_encoder_()

        if freeze_encoder:
            self.freeze_encoder()
        #Decoders 1-10
        #First 5 estimate regular depth maps using ordinal loss and SID algorithm
        self.d_1 = Decoder(in_channels=1056, num_wsm_layers=0, DORN=True, id=1, quant=self.quantizers)
        # self.d_2 = Decoder(in_channels=1056, num_wsm_layers=1, DORN=True, id=2, quant=self.quantizers)
        # self.d_3 = Decoder(in_channels=1056, num_wsm_layers=2, DORN=True, id=3, quant=self.quantizers)
        # self.d_4 = Decoder(in_channels=1056, num_wsm_layers=3, DORN=True, id=4, quant=self.quantizers)
        # self.d_5 = Decoder(in_channels=1056, num_wsm_layers=4, DORN=True, id=5, quant=self.quantizers)
        
        #Remaining 5 estimate relative depth maps using ALS
        #self.d_6 = Decoder(in_channels=1056, num_wsm_layers=0, DORN=False, id=6, quant=self.quantizers)
        #self.d_7 = Decoder(in_channels=1056, num_wsm_layers=1, DORN=False, id=7, quant=self.quantizers)
        #self.d_8 = Decoder(in_channels=1056, num_wsm_layers=2, DORN=False, id=8, quant=self.quantizers)
        #self.d_9 = Decoder(in_channels=1056, num_wsm_layers=3, DORN=False, id=9, quant=self.quantizers)
        # self.d_10 = Decoder(in_channels=1056, num_wsm_layers=4, DORN=False, id=10, quant=self.quantizers)

        self.weight_layer = Weights(vector_sizes=[1,1,1,1,0,0,0,0], use_cuda=use_cuda, relative_only=False)

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
            

    def forward(self, x):
        #print(self.weight_layer.weight_list)
        #encoder propagation
        # print("Encoder input: {0}".format(x))
        x = self.encoder.conv_e1(x)
        # print("Encoder layer 1: {0}".format(x))
        #print("Encoder layer 1 weights: {0}".format(self.encoder.conv_e1.weight.grad))
        x = self.encoder.max_e1(x)
        # print("Encoder layer 2: {0}".format(x))
        x = self.encoder.dense_e2(x)
        # print("Encoder layer 3: {0}".format(x))
        x = self.encoder.pad_br(x)
        # print("Encoder layer 4: {0}".format(x))
        x = self.encoder.trans_e2(x)
        # print("Encoder layer 5: {0}".format(x))
        x = self.encoder.dense_e3(x)
        # print("Encoder layer 6: {0}".format(x))
        x = self.encoder.pad_br(x)
        # print("Encoder layer 7: {0}".format(x))
        x = self.encoder.trans_e3(x)
        # print("Encoder layer 8: {0}".format(x))
        x = self.encoder.dense_e4(x)
        # print("Encoder layer 9: {0}".format(x))
        x = self.encoder.pad_br(x)
        # print("Encoder layer 10: {0}".format(x))
        x = self.encoder.trans_e4(x)
        # print("Encoder output: {0}".format(x))
        #according to the authors, optimal performance is reached with decoders
        #1,6,7,8,9
        ## print("Encoder output: {0}".format(x))
        #print("NaN encoder output: {0}".format(torch.isnan(x).any()))
        if use_cuda:
            x.cuda()
        x_d1, ord_labels = None, None
        x_d1, ord_labels = self.d_1(x)#regular
        #B,C,H,W = x_d1.size()
        #print("NaN after decoder: {0}".format(torch.isnan(torch.div(x_d1,cp.quick_gm(x_d1.view(B,H*W,1), H).expand(B,H*W).view(B,1,H,W))).any()))
        #print(x_d1)
        #x_d6 = self.d_6(x)#relative
        #x_d7 = self.d_7(x)#relative
        #x_d8 = self.d_8(x)#relative
        #x_d9 = self.d_9(x)#relative
        # print("D1 output before decomposition: {0}".format(x_d1))
        #get fine-detail maps for each depth map
        #print(x_d1,x_d6, x_d7, x_d8, x_d9)
        #B,C,H,W = x_d6.size()
        if not (x_d1 is None):
            B,C,H,W = x_d1.size()
        #print("d6 output < 0: {0}".format((x_d6 < 0).any()))
        f_d1 = cp.decomp(torch.div(x_d1,cp.quick_gm(x_d1.view(B,H*W,1), H).expand(B,H*W).view(B,1,H,W)), 3)[::-1]
        #print("NaN after decomp: {0}".format(torch.isnan(f_d1[0]).any()))
        #f_d6 = cp.decomp(x_d6, 3, relative_map=True)[::-1]
        #check = [(x<0).any() for x in f_d6]
        #print("f_d6 < 0: {0}".format(True in check))
        #f_d7 = cp.decomp(x_d7, 4, relative_map=True)[::-1]
        #f_d8 = cp.decomp(x_d8, 5, relative_map=True)[::-1]
        #f_d9 = cp.decomp(x_d9, 6, relative_map=True)[::-1]
        #print(f_d1, f_d6, f_d7, f_d8, f_d9)
        #bring into matrix form
        #y_hat = cp.relative_fine_detail_matrix([f_d1, f_d6, f_d7, f_d8, f_d9], use_cuda)
        y_hat = cp.relative_fine_detail_matrix([f_d1], use_cuda)
        #print("NaN after logspace switch: {0}".format(torch.isnan(y_hat[0]).any()))
        #self.weight_layer.print_grads()
        #self.weight_layer.print_grads()
        #print(list(self.weight_layer.parameters()))
        #print("yhat before fine detail part:\n{0}".format(y_hat))
        y_hat = self.weight_layer(y_hat)
        #print("NaN net output: {0}".format(torch.isnan(y_hat[0]).any()))
        #print(self.weight_layer.weight_list)
        return y_hat, x_d1, ord_labels

class Decoder(nn.Module):
    def __init__(self, in_channels, num_wsm_layers, DORN, id, quant):
        super(Decoder, self).__init__()

        """Code to assemble decoder block"""
        assert num_wsm_layers < 5 and num_wsm_layers >= 0
        self.id = id
        self.dense_layer = torchvision.models.densenet._DenseBlock(24, 528, 8, 48, 0.0, True)
        self.wsm_block = _make_wsm_layers_(num_wsm_layers)
        self.conv1 = nn.Conv2d(in_channels=_wsm_output_planes(id), out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=_wsm_output_planes(id), out_channels=180, kernel_size=1)
        self.ord_layer = Ordinal_Layer(id, DORN, quant)

    def forward(self, x):

        x = self.dense_layer(x)
        #print(x.shape)
        x = self.wsm_block(x)
        #print(x.shape)
        if self.id > 5:
            x = self.conv1(x)#make feature map have only one channel
        if self.id == 1:
            x = self.conv2(x)
        x = self.ord_layer(x)
        #print(x)
        return x
class WSMLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, layer_id):
        super(WSMLayer, self).__init__()

        """Code to assemble wsm block"""

        #deconvolution, increases size to 2 x input size
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            )
        
        #Compute remaining input channels
        kernel_in = int(in_channels/4)
        wsm_in = int(in_channels/8)

        #5 1x1x convolutions
        self.conv1_1 = nn.Conv2d(in_channels, kernel_in, 1)
        self.conv1_2 = nn.Conv2d(in_channels, kernel_in, 1)
        self.conv1_3 = nn.Conv2d(in_channels, kernel_in, 1)
        self.conv1_4 = nn.Conv2d(in_channels, wsm_in, 1)
        self.conv1_5 = nn.Conv2d(in_channels, wsm_in, 1)
        
        #WSM layer outputs concatenated with convolution layer output
        self.conv2_1 = nn.Conv2d(kernel_in, kernel_in, 3, padding=1)
        self.conv2_2 = nn.Conv2d(kernel_in, kernel_in, 5, padding=2)
        self.wsm_wx3 = _make_wsm_vertical_(wsm_in, wsm_in, (3,kernel_size), (1,stride))
        self.wsm_3xh = _make_wsm_horizontal_(wsm_in, wsm_in, (kernel_size,3), (stride,1))

        #Additional components (not from paper)
        self.id = layer_id

        #This makes sure wsm blocks can be cascaded
        if(self.id == 1):
            raw = 1680
        else:
            raw = int(2*in_channels)

        self.input_adjustment_layer = nn.Conv2d(raw, in_channels, 1)        

    def forward(self, x):
        # # print(self.id)
        # # print(x.shape)
        x = self.input_adjustment_layer(x)
        out1 = self.deconv1(x)
        # # print(out1.shape)
        #first conv block
        out1_1 = self.conv1_1(out1)
        out1_2 = self.conv1_2(out1)
        out1_3 = self.conv1_3(out1)
        out1_4 = self.conv1_4(out1)
        out1_5 = self.conv1_5(out1)
        #second conv and wsm block
        out2_1 = self.conv2_1(out1_2)
        out2_2 = self.conv2_2(out1_3)
        out_wsm_wx3 = self.wsm_wx3(out1_4)
        out_wsm_3xh = self.wsm_3xh(out1_5)
        completion_horizontal=out_wsm_wx3
        completion_vertical=out_wsm_3xh
        #complete wsm layer outputs 

        completion_horizontal = completion_horizontal.repeat(1,1,1,completion_horizontal.shape[2])
        completion_vertical = completion_vertical.repeat(1,1, completion_vertical.shape[3],1)
     
        """
        # print(out1_1.shape)
        # print(out2_1.shape)
        # print(out2_2.shape)
        # print(completion_vertical.shape)
        # print(completion_horizontal.shape)
        """
        #concatenate output of wsm layers and convolution layers
        cat = torch.cat((out1_1, out2_1, out2_2, completion_vertical, completion_horizontal),1)
        ## print(cat.shape)
        return cat
class Ordinal_Layer(nn.Module):
    def __init__(self, decoder_id, DORN, quantizer):
        super(Ordinal_Layer, self).__init__()
        self.quant = quantizer
        self.id = decoder_id-3
        self.dorn = DORN

    def sparse_comparison_v1(self, d_3):
        B,C,H,W = d_3.size()
        size = H*W
        reshaped_d_3 = d_3.view(B, C, size)
        inverse_reshaped = torch.pow(reshaped_d_3, -1)
        #sparse_m = torch.zeros(B, size, size)

        #for i in range(B):
        sparse_m = torch.matmul(reshaped_d_3.view(B, size, C), inverse_reshaped).view(B, size, size)

        depth_labels = torch.zeros(B, size, size, 40)
        relative_depth_map = self.LloydQuantization(depth_labels, sparse_m)
        ## print(relative_depth_map.shape)
        return relative_depth_map

    def sparse_comparison_id(self, dn, dn_1):
        B,C,H,W = dn.size()
        H_1,W_1 = int(H/2),int(W/2)
        dn = dn.view(B,H,W)
        test = []
        #sparse_m = torch.zeros(B,H*W,H_1*W_1)
        #for b in range(B):
        for index_row in range(H):
                for index_col in range(W):
                    index_resized_row = np.floor(index_row/2)
                    index_resized_col = np.floor(index_col/2)
                    index_row_start = int(min(max(index_resized_row, 0), dn_1.shape[2]-3))
                    index_row_end = index_row_start+2
                    index_col_start = int(min(max(index_resized_col, 0), dn_1.shape[3]-3))
                    index_col_end = index_col_start+3
                    comparison_area = cp.get_resized_area(index_row_start, index_row_end, index_col_start, index_col_end, dn_1) 
                    #print((dn[:, index_row, index_col].view(B,1,1)*torch.pow(comparison_area,-1))[0][0])
                    #return
                    test.append(dn[:, index_row, index_col].view(B,1,1)*torch.pow(comparison_area,-1))
                    #print(tmp.shape)
                    #sparse_m[:, index_row*index_col, :] = tmp[:, 0, :]
        sparse_m = torch.cat(test,1)
        depth_labels = torch.zeros(B,H*W,H_1*W_1, 40) 
        relative_depth_map = self.LloydQuantization(depth_labels, sparse_m, id=self.id)
        ## print(relative_depth_map.shape)
        return relative_depth_map

    def LloydQuantization(self, labels, relative_depths,  id=3):
        N, C, W, H = labels.size()
        
        if id == 3:
            for i in range(40):
                labels[:,:,:,i] = (relative_depths >= self.quant.depth_ratio_008_008_quant[i][0])

            indices = torch.flatten(torch.sum(labels,3))
            relative_depths = torch.flatten(relative_depths)

            for i in range(relative_depths.shape[0]):
                relative_depths[i] = self.quant.depth_ratio_008_008_quant_inv[int(indices[i])][0]

            return  relative_depths.view(N, C, W)

        elif id > 3:
            q, inv = self.quant.get_with_id(id)

            for i in range(40):
                labels[:,:,:,i] = (relative_depths >= q[i][0])

            indices = torch.flatten(torch.sum(labels,3))
            relative_depths = torch.flatten(relative_depths)
            for i in range(relative_depths.shape[0]):
                relative_depths[i] = inv[int(indices[i])][0]
            return relative_depths.view(N, C, W)
      
    def DornOrdinalRegression(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ## print("regression input tensor size: "+str(x.size()))
        ord_num = C // 2
        """
        replace iter with matrix operation
        fast speed methods
        """
        #implementation from DORN paper
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e4)  # prevent nans
        C = C.double()

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)

        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        #decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)

        return decode_c, ord_c1

    def forward(self, x):
        #print(x)
        if self.dorn:
            depth, labels = self.DornOrdinalRegression(x)
            return (depth, labels) 
        else:
            if self.id == 3:
                #use regular comparison matrix
                x = self.sparse_comparison_v1(x)
                x = cp.quadratic_als(x, cuda=x.is_cuda, n=3)
                return x 

            elif self.id == 4:
                #use comparison scheme described in paper
                dn = x 
                dn_1 = cp.resize(dn, self.quant.get_size_id(self.id-1))
                #print("D7 input as d_n: {0}".format(dn))
                #print("D7 input as d_n-1: {0}".format(dn_1))
                x = self.sparse_comparison_id(dn, dn_1)
                #print("D7 output after comparison: {0}".format(x))
                filled_map = cp.alternating_least_squares(sparse_m=x, n=4, limit=100, cuda=x.is_cuda)
                # print(filled_map.shape)
                # print("D7 done.")
                return filled_map

            elif self.id > 4:
                #for efficiency depth maps are split into 16x16 and 8x8
                dn = x 
                dn_1 = cp.resize(dn, self.quant.get_size_id(self.id-1))
                #print("D8+9 input as d_n: {0}".format(dn))
                #print("D8+9 input as d_n-1: {0}".format(dn_1))
                dn_pages, dn_1_pages = cp.split_matrix(dn, dn_1) #two lists of split pages (same length) from dn and dn_1
                zipped = zip(dn_pages,dn_1_pages)
                sparse_pages = [self.sparse_comparison_id(z[0], z[1]) for z in zipped]
                als_filled_pages = [cp.alternating_least_squares(sparse, n=4, limit=100, cuda=x.is_cuda) for sparse in sparse_pages]
                full_map = cp.reconstruct(als_filled_pages)
                # print(full_map.shape)
                # print("D{0} done.".format(self.id+3))
                return full_map            
class Quantization():
    def __init__(self):
        """
        Quantizing levels are taken from the creators paper und held in this class for
        easier access.
        """
        quant_8 = scipy.io.loadmat("depth_ratio_008_008_quant.mat")
        quant_16 = scipy.io.loadmat("depth_ratio_016_016_quant.mat")
        quant_32 = scipy.io.loadmat("depth_ratio_032_032_quant.mat")
        quant_64 = scipy.io.loadmat("depth_ratio_064_064_quant.mat")
        quant_128 = scipy.io.loadmat("depth_ratio_128_128_quant.mat")

        self.depth_ratio_008_008_quant = quant_8['depth_ratio_008_008_quant']
        self.depth_ratio_008_008_quant_inv = quant_8['depth_ratio_008_008_quant_inv']
        self.depth_ratio_016_016_quant = quant_16['depth_ratio_016_016_quant']
        self.depth_ratio_016_016_quant_inv = quant_16['depth_ratio_016_016_quant_inv']
        self.depth_ratio_032_032_quant = quant_32['depth_ratio_032_032_quant']
        self.depth_ratio_032_032_quant_inv = quant_32['depth_ratio_032_032_quant_inv']
        self.depth_ratio_064_064_quant = quant_64['depth_ratio_064_064_quant']
        self.depth_ratio_064_064_quant_inv = quant_64['depth_ratio_064_064_quant_inv']
        self.depth_ratio_128_128_quant = quant_128['depth_ratio_128_128_quant']
        self.depth_ratio_128_128_quant_inv = quant_128['depth_ratio_128_128_quant_inv']
    
    def get_with_id(self, id):
        if id == 3:
            return self.depth_ratio_008_008_quant, self.depth_ratio_008_008_quant_inv
        elif id == 4:
            return self.depth_ratio_016_016_quant, self.depth_ratio_016_016_quant_inv
        elif id == 5:
            return self.depth_ratio_032_032_quant, self.depth_ratio_032_032_quant_inv
        elif id == 6:
            return self.depth_ratio_064_064_quant, self.depth_ratio_064_064_quant_inv
        elif id == 7:
            return self.depth_ratio_128_128_quant, self.depth_ratio_128_128_quant_inv
    
    def get_size_id(self, id):
        if id == 3:
            return 8
        elif id == 4:
            return 16
        elif id == 5:
            return 32
        elif id == 6:
            return 64
        elif id == 7:
            return 128
class Weights(nn.Module):
    def __init__(self, vector_sizes, use_cuda, relative_only):
        super(Weights, self).__init__()
        self.use_cuda = use_cuda
        self.relative_only = relative_only
        if self.use_cuda:
            self.d0 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[0],1)), dim=0).cuda())
            self.f1 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[1],1)), dim=0).cuda())
            self.f2 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[2],1)), dim=0).cuda())
            self.f3 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[3],1)), dim=0).cuda())
            self.f4 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[4],1)), dim=0).cuda())
            self.f5 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[5],1)), dim=0).cuda())
            self.f6 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[6],1)), dim=0).cuda())
            self.f7 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[7],1)), dim=0).cuda())
        else:
            self.d0 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[0],1)), dim=0))
            self.f1 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[1],1)), dim=0))
            self.f2 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[2],1)), dim=0))
            self.f3 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[3],1)), dim=0))
            self.f4 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[4],1)), dim=0))
            self.f5 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[5],1)), dim=0))
            self.f6 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[6],1)), dim=0))
            self.f7 = nn.Parameter(nn.functional.softmax(torch.ones((vector_sizes[7],1)), dim=0))
        self.weight_list = [self.d0, self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7]
        
        for weight_vector in self.weight_list:
            if weight_vector.shape[0] == 0:
                weight_vector.requires_grad = False

    def update(self, weight_index, lr, gradient):
        self.weight_list[weight_index] = self.weight_list[weight_index] - lr * gradient
    
    def get(self, index):
        return self.weight_list[index]

    def _make_weightvector_list_(self, sizes, use_cuda=False):

        if use_cuda:
            return [torch.nn.Parameter(nn.functional.softmax(torch.ones((size,1)), dim=0).cuda()) for size in sizes]
        
        return [torch.nn.Parameter(nn.functional.softmax(torch.ones((size,1)), dim=0)) for size in sizes]
    
    def print_grads(self):
        for weight in self.weight_list:
            if not weight.shape[0] == 0 and not weight.grad is None:
                print(torch.isnan(weight.grad).any())
    
    def forward(self, x):
        return cp.make_pred(self.weight_list, x, self.use_cuda, self.relative_only)

def _make_wsm_vertical_(in_channels, out_channels, kernel_size, stride):
    """Stride has to be chosen in a way that only one convolution is performed
       The output is a compressed feature column.
    """
    wsm_module = nn.Sequential(
        nn.ZeroPad2d((0,0,1,1)),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )

    return wsm_module

def _make_wsm_horizontal_(in_channels, out_channels, kernel_size, stride):
    """Stride has to be chosen in a way that only one convolution is performed
       Padding different for horizontal and vertical. Output is a compressed feature
    """
    wsm_module = nn.Sequential(
        nn.ZeroPad2d((1,1,0,0)),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )

    return wsm_module

def _make_encoder_():
    denseNet = torchvision.models.densenet
    encoder = _get_denseNet_Components(denseNet)

    return encoder

def _get_denseNet_Components(denseNet):

    encoder = nn.Module()
    encoder.conv_e1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, out_channels=48, padding=3)
    encoder.max_e1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    encoder.dense_e2 = denseNet._DenseBlock(6, 48, 57, 48, 0.0, True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=336, num_output_features=96)
    encoder.dense_e3 = denseNet._DenseBlock(12, 96, 29, 48, 0.0, True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=672, num_output_features=192)
    encoder.dense_e4 = denseNet._DenseBlock(36, 192, 15, 48, 0.0, True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=1920, num_output_features=528)
    encoder.pad_br = nn.ZeroPad2d((0,1,0,1))

    return encoder

def _make_wsm_layers_(num_of_layers):
    
    wsm_d1 = WSMLayer(832, 16, 16, 1)
    wsm_d2 = WSMLayer(416, 32, 32, 2)
    wsm_d3 = WSMLayer(208, 64, 64, 3)
    wsm_d4 = WSMLayer(104, 128, 128, 4)

    wsm_block = nn.Sequential()
    if num_of_layers > 0:
        wsm_block.add_module("WSM_1",wsm_d1)
    if num_of_layers > 1:
        wsm_block.add_module("WSM_2",wsm_d2)
    if num_of_layers > 2: 
        wsm_block.add_module("WSM_3",wsm_d3)
    if num_of_layers > 3:
        wsm_block.add_module("WSM_4",wsm_d4)

    return wsm_block

def _wsm_output_planes(decoder_id):
    if decoder_id==6 or decoder_id == 1:
        return 1680
    elif decoder_id==7:
        return 832
    elif decoder_id==8:
        return 416
    elif decoder_id==9:
        return 208
    elif decoder_id==10:
        return 104
    else:
        return 1

def debug(container, id):
    print("Found {0} fine detail maps in decoder {1} output.".format(len(container), id))
    print("Shapes:")
    for t in container:
        print(t.shape)
    print("Nans -> {0}".format(cp.find_nans(container)))
    print("\n")


if __name__ == "__main__":
    inp = torch.randn((4, 3, 226, 226))

    model = DepthEstimationNet()
    r,_,_ = model(inp)
    print(r)


