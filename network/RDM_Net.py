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

        """Code to assemble decoder blocks
        
        
        """
    
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
    def __init__(self, in_channels):
        super(WSMLayer, self).__init__()

        """Code to assemble decoder blocks
        
        
        """
    
    def forward(self, x):

          """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

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
    
    image = torch.rand((16,3,226,226))
    model = DepthEstimationNet("")
    print(model)
    # print(image)

    pretrained = model(image)
    print(pretrained)


