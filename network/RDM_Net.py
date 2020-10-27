import torch
import torch.nn as nn
import torchvision


class DepthEstimationNet(nn.Module):
    def __init__(self, block, layers_encoder, layers_decoder):
        super(DepthEstimationNet, self).__init__()

        self.encoder = _make_encoder_()
    
def _make_encoder_():
    encoder = _get_denseNet_Components()

    return encoder

def _get_denseNet_Components():
    denseNet = torchvision.models.densenet
    encoder = nn.Module()
    encoder.conv_e1 = nn.Conv2d(in_channels=3, kernel_size=7, stride=2, out_channels=3)
    encoder.max_e1 = nn.MaxPool2d(kernel_size=3, stride=2)
    encoder.dense_e2 = denseNet._DenseLayer(96, 48, bn_size=57, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e2 = denseNet._Transition(num_input_features=384, num_output_features=192)
    encoder.dense_e3 = denseNet._DenseLayer(192, 48, bn_size=29, drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e3 = denseNet._Transition(num_input_features=768, num_output_features=384)
    encoder.dense_e4 = denseNet._DenseLayer(384, 48, bn_size=15,  drop_rate= 0.0, memory_efficient=True)
    encoder.trans_e4 = denseNet._Transition(num_input_features=2112, num_output_features=1056)

    return encoder

class Decoder(nn.Module):