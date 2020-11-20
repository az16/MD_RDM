import torch
import torch.nn as nn
import torchvision


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
    """
    encoder.dense_e2 = denseNet._DenseLayer(96, 48, bn_size=57, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=48, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseLayer(192, 48, bn_size=29, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=48, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseLayer(384, 48, bn_size=15, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=48, num_output_features=1056)
    """

    encoder.dense_e2 = denseNet._DenseBlock(6, 96, 57, 48, 0.0, True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=384, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseBlock(12, 192, 29, 48, 0.0, True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=768, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseBlock(36, 384, 15, 48, 0.0, True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=2112, num_output_features=1056)
    encoder.pad_tl = nn.ZeroPad2d((1,0,1,0))

    return encoder

class Decoder(nn.Module):
    def __init__(self, in_channels, num_wsm_layers, use_als_layer, DORN, Lloyd):
        super(Decoder, self).__init__()

        """Code to assemble decoder block"""
        assert num_wsm_layers < 5 and num_wsm_layers >= 0

        self.dense_layer = torchvision.models.densenet._DenseBlock(24, 1056, 8, 48, 0.0, True)
        self.wsm_block = _make_wsm_layers_(num_wsm_layers)
        if DORN:
            self.ord_layer = Ordinal_Layer()
        elif Lloyd:
            self.ord_layer = Lloyd_Quantization_Layer()
    
    def forward(self, x):

        x = self.dense_layer(x)
        #print(x.shape)
        x = self.wsm_block(x)
        #print(x.shape)
        x = self.ord_layer(x)

        return x


class ALSLayer(nn.Module):
    def __init__(self, in_channels):
        super(ALSLayer, self).__init__()

        """Code to assemble decoder blocks
        
        
        """
    
    def forward(self, x):

          """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """


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
    
class Lloyd_Quantization_Layer(nn.Module):
    def __init__(self):
        super(Lloyd_Quantization_Layer, self).__init__()

    def forward(self, x):
        print(x)
        return x


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
    
    encoder_output = torch.randn((1, 1056, 8, 8))
    decoder_block  = Decoder(1056, num_wsm_layers=2, use_als_layer=False, DORN=False,Lloyd=True)
    print(decoder_block(encoder_output).shape)
    



