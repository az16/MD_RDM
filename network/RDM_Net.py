import torch
import torch.nn as nn
import torchvision
#import network.transforms as t
import numpy as np
import scipy.io
import computations as cp
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
    def __init__(self, path):
        super(DepthEstimationNet, self).__init__()
        self.quantizers = Quantization()
        self.encoder = _make_encoder_()
    


    def forward(self, x):
        #encoder propagation
        print("\nEncoder shapes after each layer: \n")
        x = self.encoder.conv_e1(x)
        print(x.shape)
        x = self.encoder.max_e1(x)
        print(x.shape)
        x = self.encoder.dense_e2(x)
        print(x.shape)
        x = self.encoder.trans_e2(x)
        x = self.encoder.pad_tl(x)
        print(x.shape)
        x = self.encoder.dense_e3(x)
        print(x.shape)
        x = self.encoder.trans_e3(x)
        x = self.encoder.pad_tl(x)
        print(x.shape)
        x = self.encoder.dense_e4(x)
        print(x.shape)
        x = self.encoder.trans_e4(x)
        x = self.encoder.pad_tl(x)
        print(x.shape)
        print("\n")

        return x
        
    

def _make_encoder_():
    denseNet = torchvision.models.densenet
    encoder = _get_denseNet_Components(denseNet)

    return encoder

def _get_denseNet_Components(denseNet):

    encoder = nn.Module()
    encoder.conv_e1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, out_channels=96, padding=3)
    encoder.max_e1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    encoder.dense_e2 = denseNet._DenseBlock(6, 96, 57, 48, 0.0, True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=384, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseBlock(12, 192, 29, 48, 0.0, True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=768, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseBlock(36, 384, 15, 48, 0.0, True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=2112, num_output_features=1056)
    encoder.pad_tl = nn.ZeroPad2d((1,0,1,0))

    return encoder

class Decoder(nn.Module):
    def __init__(self, in_channels, num_wsm_layers, DORN, id, quant):
        super(Decoder, self).__init__()

        """Code to assemble decoder block"""
        assert num_wsm_layers < 5 and num_wsm_layers >= 0

        self.dense_layer = torchvision.models.densenet._DenseBlock(24, 1056, 8, 48, 0.0, True)
        self.wsm_block = _make_wsm_layers_(num_wsm_layers)
        if DORN:
            self.ord_layer = Ordinal_Layer()
        else:
            self.ord_layer = Lloyd_Quant(id, quant)  


    def forward(self, x):

        x = self.dense_layer(x)
        #print(x.shape)
        x = self.wsm_block(x)
        #print(x.shape)
        x = self.ord_layer(x)

        return x

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
        # print(self.id)
        # print(x.shape)
        x = self.input_adjustment_layer(x)
        out1 = self.deconv1(x)
        # print(out1.shape)
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
        print(out1_1.shape)
        print(out2_1.shape)
        print(out2_2.shape)
        print(completion_vertical.shape)
        print(completion_horizontal.shape)
        """
        #concatenate output of wsm layers and convolution layers
        cat = torch.cat((out1_1, out2_1, out2_2, completion_vertical, completion_horizontal),1)
        print(cat.shape)

        return cat
        
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

def wsm_test(image):
    """Stride has to be chosen in a way that only one convolution is performed
        Padding different for horizontal and vertical. Output is a compressed feature
    """
    wsm_module = nn.Sequential(
        nn.ZeroPad2d((1,1,0,0)),
        nn.Conv2d(1, 1, (5,3), (5,1))
        )
    
    return wsm_module(image)

#From DORN paper
class Ordinal_Layer(nn.Module):
    def __init__(self):
        super(Ordinal_Layer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        #print("regression tensor size"+str(x.size()))
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

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)

        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W)
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1

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

def sparse_comparison_v1(d_3, quant):
    reshaped_d_3 = torch.reshape(d_3, (1, 1, d_3.shape[2]*d_3.shape[3]))
    sparse_m = torch.empty(d_3.shape[2], d_3.shape[3], d_3.shape[2]*d_3.shape[3])

    for i in range(d_3.shape[2]):
        for j in range(d_3.shape[3]):
            sparse_m[i][j][:] = reshaped_d_3/d_3[0][0][i][j]

    r_sparse = torch.reshape(sparse_m, (1, 64*64))
    tmp = torch.empty(r_sparse.shape)

    for i in range(r_sparse.shape[1]):
        distances = [abs(x[0]-r_sparse[0][i]) for x in quant.depth_ratio_008_008_quant]
        index = distances.index(min(distances))+1
        tmp[0][i] = quant.depth_ratio_008_008_quant_inv[index][0]
    
    relative_depth_map = torch.reshape(tmp, (64,64))  
    relative_depth_map = cp.fill_sparse_R3(relative_depth_map)
       
    print(relative_depth_map)  
    return relative_depth_map
    

def sparse_comparison_id(dn, dn_1, quant, id):
    size_dn_1, size_dn = cp.get_size(id)
    sparse_m = torch.empty(dn.shape[2], dn.shape[3], size_dn_1**2)
    q = torch.reshape(dn, (1, size_dn**2))
    p = torch.reshape(dn_1, (1, size_dn_1**2))
    for index_row in range(dn.shape[2]):
            for index_col in range(dn.shape[3]):
                index_resized_row = np.floor(index_row/2)
                index_resized_col = np.floor(index_col/2)
                index_row_start = int(min(max(index_resized_row, 0), dn_1.shape[2]-3))
                index_row_end = index_row_start+2
                index_col_start = int(min(max(index_resized_col, 0), dn_1.shape[3]-3))
                index_col_end = index_col_start+3
                comparison_area = cp.get_resized_area(index_row_start, index_row_end, index_col_start, index_col_end, dn_1)
                sparse_m[index_row][index_col][:] = comparison_area[0][0] / dn[0][0][index_row][index_col]
               
    quantizer, inverse = quant.get_with_id(id)
    r_sparse = torch.reshape(sparse_m, (size_dn**2,size_dn_1**2))
    #cp.fill_sparse_Rn(r_sparse, q, p, 100) 
    """
    tmp = torch.empty(r_sparse.shape)

    for i in range(r_sparse.shape[1]):
        distances = [abs(x[0]-r_sparse[0][i]) for x in quantizer]
        index = distances.index(min(distances))+1
        tmp[0][i] = inverse[index][0]
    
    relative_depth_map = torch.reshape(tmp, (dn.shape[2]*dn.shape[3], size_dn_1**2))    
    """
        
if __name__ == "__main__":
    #encoder test lines
    """
    print("Encoder test\n")
    image = torch.randn((16,3,226,226))
    model = DepthEstimationNet("")
    print("Image\n")
    print(image.shape)
    #print(model)
    pretrained = model(image)
    print("Encoder result\n")
    print(pretrained.shape)
    """
    """
    print("WSMLayer test\n")
    #wsm test lines
    print("Test image\n")
    wsm_test_image = torch.rand((1,1,5,5))
    print(wsm_test_image)
    compressed_horizontal = wsm_test(wsm_test_image)
    print("WSM compressed feature\n")
    print(compressed_horizontal)
    """

    #decoder test lines
    
    encoder_output = torch.randint(1,10,(1, 1, 8, 8))
    encoder_output2 = torch.randint(1,10,(1, 1, 16, 16))
    quant = Quantization()
    #decoder_block_1 = Decoder(1056, num_wsm_layers=0, DORN=False, id=0, quant=quant)
    #x = decoder_block_1(encoder_output)
    #print(x)
    #sparse_comparison_v1(encoder_output, quant)
    # print(labels)
    # print(labels.shape)
    sparse_comparison_id(encoder_output2, encoder_output, quant, 4)
    #print(get_resized_area(0,3,0,3,encoder_output).shape)


    # test = Quantization()
    # print(len(test.depth_ratio_008_008_quant))
    # print(len(test.depth_ratio_008_008_quant_inv))
    # print(test.depth_ratio_016_016_quant[-1])
    # print(test.depth_ratio_032_032_quant[-1])

    #image = torch.randn((16,3,226,226))
    #print(len(image[1]))


    



