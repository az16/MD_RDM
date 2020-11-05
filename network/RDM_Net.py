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
    wsm_d1 = WSMLayer(1664, 1664, 416, 208, 16)
    wsm_d2 = WSMLayer(832, 832, 208, 104, 32)
    wsm_d3 = WSMLayer(416, 416, 104, 52, 64)
    wsm_d4 = WSMLayer(208, 208, 52, 26, 128)

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
    def __init__(self, in_channels, out_channels, kernel_in, wsm_in, size):
        super(WSMLayer, self).__init__()

        """Code to assemble wsm block"""


        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(kernel_in, kernel_in, 1),
            nn.Conv2d(kernel_in, kernel_in, 1),
            nn.Conv2d(kernel_in, kernel_in, 1),
            nn.Conv2d(kernel_in, kernel_in, 1),
            nn.Conv2d(kernel_in, kernel_in, 1)
        )

    def forward(self, x):

          """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """


def wsm_vertical(x):
    #pad top and bottom
    m = nn.ZeroPad2d((0,0,1,1)) 
    padded = m(x)
    #Wx3 conv
    conv = nn.Conv2d(1, 1, (3,5), (1,5))
    conv_col = conv(padded)
    replicated = conv_col
    for i in range (5):
        replicated = torch.cat((replicated, conv_col),0)
        i+=1
    return replicated

def wsm_horizontal(x):
    m = nn.ZeroPad2d((1,1,0,0))
    padded = m(x)
    #3xH conv
    conv = nn.Conv2d(1, 1, (5,3), (5,1))
    conv_row = conv(padded)
    replicated = conv_row
    for i in range (5):
        replicated = torch.cat((replicated, conv_row),0)
        i+=1
    return replicated


class Ordinal_Layer(nn.Module):
    def __init__(self, in_channels):
        super(Ordinal_Layer, self).__init__()

        """Code to assemble decoder blocks
        
        
        """
    
    def forward(self, x):

          """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

if __name__ == "__main__":
    
    image = torch.rand((1,1,5,5))
    #model = DepthEstimationNet("")
    #print(model)
    print(image)
    print(wsm_horizontal(image))
    print(wsm_vertical(image))

    # pretrained = model(image)
    # print(pretrained)


