import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io
import network.computations as cp
from PIL import Image

class BaseModel(nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
        """
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

        #Quantizers for Lloyd quantization
        self.quantizers = Quantization()
        #Encoder part
        self.encoder = _make_encoder_()
        #Decoders 1-10
        #First 5 estimate regular depth maps using ordinal loss and SID algorithm
        self.d_1 = Decoder(in_channels=1056, num_wsm_layers=0, DORN=True, id=1, quant=self.quantizers)
        # self.d_2 = Decoder(in_channels=1056, num_wsm_layers=1, DORN=True, id=2, quant=self.quantizers)
        # self.d_3 = Decoder(in_channels=1056, num_wsm_layers=2, DORN=True, id=3, quant=self.quantizers)
        # self.d_4 = Decoder(in_channels=1056, num_wsm_layers=3, DORN=True, id=4, quant=self.quantizers)
        # self.d_5 = Decoder(in_channels=1056, num_wsm_layers=4, DORN=True, id=5, quant=self.quantizers)
        
        #Remaining 5 estimate relative depth maps using ALS
        self.d_6 = Decoder(in_channels=1056, num_wsm_layers=0, DORN=False, id=6, quant=self.quantizers)
        self.d_7 = Decoder(in_channels=1056, num_wsm_layers=1, DORN=False, id=7, quant=self.quantizers)
        self.d_8 = Decoder(in_channels=1056, num_wsm_layers=2, DORN=False, id=8, quant=self.quantizers)
        self.d_9 = Decoder(in_channels=1056, num_wsm_layers=3, DORN=False, id=9, quant=self.quantizers)
        # self.d_10 = Decoder(in_channels=1056, num_wsm_layers=4, DORN=False, id=10, quant=self.quantizers)

        self.weight_layer = Weights([1,5,5,5,3,2,1,0])

    def forward(self, x):
        #encoder propagation
        ## print("Encoder input: {0}".format(x))
        x = self.encoder.conv_e1(x)
        # print("Encoder layer 1: {0}".format(x))
        # print("Encoder layer 1 weights: {0}".format(self.encoder.conv_e1.weight))
        x = self.encoder.max_e1(x)
        ## print("Encoder layer 2: {0}".format(x))
        x = self.encoder.dense_e2(x)
        ## print("Encoder layer 3: {0}".format(x))
        x = self.encoder.pad_br(x)
        ## print("Encoder layer 4: {0}".format(x))
        x = self.encoder.trans_e2(x)
        ## print("Encoder layer 5: {0}".format(x))
        x = self.encoder.dense_e3(x)
        ## print("Encoder layer 6: {0}".format(x))
        x = self.encoder.pad_br(x)
        ## print("Encoder layer 7: {0}".format(x))
        x = self.encoder.trans_e3(x)
        ## print("Encoder layer 8: {0}".format(x))
        x = self.encoder.dense_e4(x)
        ## print("Encoder layer 9: {0}".format(x))
        x = self.encoder.pad_br(x)
        ## print("Encoder layer 10: {0}".format(x))
        x = self.encoder.trans_e4(x)

        #according to the authors, optimal performance is reached with decoders
        #1,6,7,8,9
        ## print("Encoder output: {0}".format(x))
        x_d1 = self.d_1(x)#regular
        x_d6 = self.d_6(x)#relative
        x_d7 = self.d_7(x)#relative
        x_d8 = self.d_8(x)#relative
        x_d9 = self.d_9(x)#relative
        # print("D1 output before decomposition: {0}".format(x_d1))
        #get fine-detail maps for each depth map
        f_d1 = cp.decompose_depth_map([], x_d1, 3)[::-1]
        f_d6 = cp.decompose_depth_map([], x_d6, 3, relative_map=True)[::-1]
        f_d7 = cp.decompose_depth_map([], x_d7, 4, relative_map=True)[::-1]
        f_d8 = cp.decompose_depth_map([], x_d8, 5, relative_map=True)[::-1]
        f_d9 = cp.decompose_depth_map([], x_d9, 6, relative_map=True)[::-1]

        #bring into matrix form
        y_hat = cp.relative_fine_detail_matrix([f_d1.cuda(), f_d6.cuda(), f_d7.cuda(), f_d8.cuda(), f_d9.cuda()])
        
        return y_hat, x_d1
class Decoder(nn.Module):
    def __init__(self, in_channels, num_wsm_layers, DORN, id, quant):
        super(Decoder, self).__init__()

        """Code to assemble decoder block"""
        assert num_wsm_layers < 5 and num_wsm_layers >= 0
        self.id = id
        self.dense_layer = torchvision.models.densenet._DenseBlock(24, 1056, 8, 48, 0.0, True)
        self.wsm_block = _make_wsm_layers_(num_wsm_layers)
        self.conv1 = nn.Conv2d(in_channels=_wsm_output_planes(id), out_channels=1, kernel_size=1)
        self.ord_layer = Ordinal_Layer(id, DORN, quant)

    def forward(self, x):

        x = self.dense_layer(x)
        ## print(x.shape)
        x = self.wsm_block(x)
        ## print(x.shape)
        if self.id > 5:
            x = self.conv1(x)#make feature map have only one channel

        x = self.ord_layer(x)

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
            raw = 2208
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

        #complete wsm layer outputs 
        completion_vertical = out_wsm_wx3
        completion_horizontal = out_wsm_3xh
        for i in range(out_wsm_wx3.shape[2]-1):
            completion_vertical = torch.cat((completion_vertical, out_wsm_wx3),3)
            i=+1
        
        for i in range(out_wsm_3xh.shape[3]-1):
            completion_horizontal = torch.cat((completion_horizontal, out_wsm_3xh),2)
            i=+1
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
        inverse_reshaped = 1/reshaped_d_3
        sparse_m = torch.zeros(B, size, size)

        for i in range(B):
            sparse_m[i] = torch.matmul(reshaped_d_3[i].T, inverse_reshaped[i])

        depth_labels = torch.zeros(B, size, size, 40)
        relative_depth_map = self.LloydQuantization(depth_labels, sparse_m)
        ## print(relative_depth_map.shape)
        return relative_depth_map

    def sparse_comparison_id(self, dn, dn_1):
        B,C,H,W = dn.size()
        H_1,W_1 = int(H/2),int(W/2)
        dn = dn.view(B,H,W)
        sparse_m = torch.zeros(B,H,W,H_1*W_1)
        for b in range(B):
            for index_row in range(H):
                    for index_col in range(W):
                        index_resized_row = np.floor(index_row/2)
                        index_resized_col = np.floor(index_col/2)
                        index_row_start = int(min(max(index_resized_row, 0), dn_1.shape[2]-3))
                        index_row_end = index_row_start+2
                        index_col_start = int(min(max(index_resized_col, 0), dn_1.shape[3]-3))
                        index_col_end = index_col_start+3
                        comparison_area = cp.get_resized_area(index_row_start, index_row_end, index_col_start, index_col_end, dn_1).cuda()
                        
                        sparse_m[b][index_row][index_col][:] = comparison_area[0][0]/dn[b][index_row][index_col]

        sparse_m = sparse_m.view(B,H*W,H_1*W_1)
        depth_labels = torch.zeros(B,H*W,H_1*W_1, 40).cuda()
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
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        # print(decode_c.shape)
        # print("D1 done.")
        return decode_c#, ord_c1

    def forward(self, x):
        if self.dorn:
            x = self.DornOrdinalRegression(x)
            return x 
        else:
            if self.id == 3:
                #use regular comparison matrix
                x = self.sparse_comparison_v1(x)
                x = cp.principal_eigen(x)
                # print(x.shape)
                # print("D6 done.")
                return x 

            elif self.id == 4:
                #use comparison scheme described in paper
                dn = x 
                dn_1 = cp.resize(dn, self.quant.get_size_id(self.id-1))
                x = self.sparse_comparison_id(dn, dn_1)
                filled_map = cp.alternating_least_squares(sparse_m=x, n=4, limit=100)
                # print(filled_map.shape)
                # print("D7 done.")
                return filled_map

            elif self.id > 4:
                #for efficiency depth maps are split into 16x16 and 8x8
                dn = x 
                dn_1 = cp.resize(dn, self.quant.get_size_id(self.id-1))
                dn_pages, dn_1_pages = cp.split_matrix(dn, dn_1) #two lists of split pages (same length) from dn and dn_1
                zipped = zip(dn_pages,dn_1_pages)
                sparse_pages = [self.sparse_comparison_id(z[0], z[1]) for z in zipped]
                als_filled_pages = [cp.alternating_least_squares(sparse, n=4, limit=100) for sparse in sparse_pages]
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
class Weights:
    def __init__(self, vector_sizes):
        self.weight_list = self._make_weightvector_list_(vector_sizes)

    def update(self, weight_index, lr, gradient):
        self.weight_list[weight_index] = self.weight_list[weight_index] - lr * gradient
    
    def get(self, index):
        return self.weight_list[index]

    def _make_weightvector_list_(self, sizes):
        return [torch.ones((size,1), requires_grad=True) for size in sizes]

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
    encoder.conv_e1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, out_channels=96, padding=3, padding_mode="reflect")
    encoder.max_e1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    encoder.dense_e2 = denseNet._DenseBlock(6, 96, 57, 48, 0.0, True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=384, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseBlock(12, 192, 29, 48, 0.0, True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=768, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseBlock(36, 384, 15, 48, 0.0, True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=2112, num_output_features=1056)
    encoder.pad_br = nn.ZeroPad2d((0,1,0,1))

    return encoder

def _make_wsm_layers_(num_of_layers):
    
    wsm_d1 = WSMLayer(1664, 16, 16, 1)
    wsm_d2 = WSMLayer(832, 32, 32, 2)
    wsm_d3 = WSMLayer(416, 64, 64, 3)
    wsm_d4 = WSMLayer(208, 128, 128, 4)

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
    if decoder_id==6:
        return 2208
    elif decoder_id==7:
        return 1664
    elif decoder_id==8:
        return 832
    elif decoder_id==9:
        return 416
    elif decoder_id==10:
        return 208
    else:
        return 1

def debug(container, id):
    print("Found {0} fine detail maps in decoder {1} output.".format(len(container), id))
    print("Shapes:")
    for t in container:
        print(t.shape)
    print("Nans -> {0}".format(cp.find_nans(container)))
    print("\n")






    



