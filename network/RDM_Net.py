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
        out1 = self.encoder.conv_e1(x)
        out2 = self.encoder.max_e1(out1)
        out3 = self.encoder.dense_e2(out2)
        out4 = self.encoder.trans_e2(out3)
        out5 = self.encoder.dense_e3(out4)
        out6 = self.encoder.trans_e3(out5)
        out7 = self.encoder.dense_e4(out6)
        out8 = self.encoder.trans_e4(out7)

        return out8
        
    

def _make_encoder_():
    denseNet = torchvision.models.densenet
    encoder = _get_denseNet_Components(denseNet)

    return encoder

def _make_wsm_layers_(num_of_layers):
    wsm_layers = []
    wsm_d1 = WSMLayer(1664, 1664, 416, 208, 16, 16)
    wsm_d2 = WSMLayer(832, 832, 208, 104, 32, 32)
    wsm_d3 = WSMLayer(416, 416, 104, 52, 64, 64)
    wsm_d4 = WSMLayer(208, 208, 52, 26, 128, 128)

    if num_of_layers > 0:
        wsm_layers.append(wsm_d1)
    if num_of_layers > 1:
        wsm_layers.append(wsm_d2)
    if num_of_layers > 2: 
        wsm_layers.append(wsm_d3)
    if num_of_layers > 3:
        wsm_layers.append(wsm_d4)

    return wsm_layers

def _get_denseNet_Components(denseNet):

    encoder = nn.Module()
    encoder.conv_e1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, out_channels=96)
    encoder.max_e1 = nn.MaxPool2d(kernel_size=3, stride=2)
    encoder.dense_e2 = denseNet._DenseLayer(96, 48, bn_size=57, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=48, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseLayer(192, 48, bn_size=29, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=48, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseLayer(384, 48, bn_size=15, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=48, num_output_features=1056)

    return encoder

class Decoder(nn.Module):
    def __init__(self, in_channels, num_wsm_layers, use_als_layer):
        super(Decoder, self).__init__()

        """Code to assemble decoder block"""
        assert num_wsm_layers < 5 and num_wsm_layers >= 0

        self.dense_layer = torchvision.models.densenet._DenseLayer(96, 48, bn_size=57, drop_rate= 0.0, memory_efficient=True)
        self.wsm_layers = _make_wsm_layers_(num_wsm_layers)
        self.ord_layer = Ordinal_Layer(self.wsm_layers[-1].out_channels)

        if use_als_layer:
            self.als_layer = ALSLayer(self.ord_layer.output_channels)
    
    def forward(self, x):

          """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

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

class WSMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_in, wsm_in, kernel_size, stride):
        super(WSMLayer, self).__init__()

        """Code to assemble wsm block"""

        #2 times deconvolution
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3)
            )

        #5 1x1x convolutions
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1_5 = nn.Conv2d(in_channels, out_channels, 1)
        
        #WSM layer outputs concatenated with convolution layer output
        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, 3)
        self.wsm_wx3 = _make_wsm_vertical_(in_channels, out_channels, (kernel_size,3), stride)
        self.wsm_wx3 = _make_wsm_horizontal_(in_channels, out_channels, (3,kernel_size), stride)
        
        

    def forward(self, x):

        out1 = self.deconv1(x)
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
        completion_vertical = []
        completion_horizontal = []
        for i in range(out_wsm_wx3.shape()[3]):
            completion_vertical = torch.cat((completion_vertical, out_wsm_wx3),0)
            i=+1
        
        for i in range(out_wsm_3xh.shape()[3]):
            completion_horizontal = torch.cat((completion_horizontal, out_wsm_3xh),0)
            i=+1

        #concatenate output of wsm layers and convolution layers
        cat = self.concat((out2_1, out2_2, completion_vertical, completion_horizontal))

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


#From DORN paper
class Ordinal_Layer(nn.Module):
    def __init__(self, input_channels):
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

if __name__ == "__main__":
    
    image = torch.rand((1,1,5,5))
    #model = DepthEstimationNet("")
    #print(model)
    print(image)
    
    # pretrained = model(image)
    # print(pretrained)


