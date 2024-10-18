import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Backbone(nn.Module):
#     def __init__(self):
#         super(Backbone, self).__init__()
#         pretrained_densenet = models.densenet121(pretrained=True)
#         self.features = nn.Sequential(*list(pretrained_densenet.features.children()))

#     def forward(self, x):
#         feat = self.features(x)
#         return feat

# class VGGPerceptualLoss(nn.Module):
#     def init(self, feature_layers=[3, 8, 15, 22], use_gpu=True):
#         super(VGGPerceptualLoss, self).init()
#         self.vgg = models.vgg16(pretrained=True).features
#         self.feature_layers = feature_layers
#         self.use_gpu = use_gpu
#         if use_gpu:
#             self.vgg = self.vgg.to(device)
#         self.vgg.eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def forward(self, x, y):
#         x_features = self.extract_features(x)
#         y_features = self.extract_features(y)
#         loss = 0
#         for xf, yf in zip(x_features, y_features):
#             loss += F.mse_loss(xf, yf)
#         return loss

#     def extract_features(self, x):
#         features = []
#         for i, layer in enumerate(self.vgg):
#             x = layer(x)
#             if i in self.feature_layers:
#                 features.append(x)
#         return features

class Backbone(nn.Module):
    def __init__(self, model_name='convnext_tiny'):
        super(Backbone, self).__init__()
        convnext = models.convnext_tiny(pretrained=True)
        self.features = nn.Sequential(*list(convnext.features.children()))
        # Add a 1x1 convolution to change the number of channels
        self.channel_adj = nn.Conv2d(768, 1024, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.channel_adj(feat)  # Adjust the number of channels
        return feat

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024):  # Keep this as is
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 6, 9, 12], use_gpu=True):
        super(ConvNeXtPerceptualLoss, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=True).features
        self.feature_layers = feature_layers
        self.use_gpu = use_gpu
        if use_gpu:
            self.convnext = self.convnext.cuda()
        self.convnext.eval()
        for param in self.convnext.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += F.mse_loss(xf, yf)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.convnext):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class GradReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.lambd * grad_output.neg()
        return grad_input, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFunction.apply(x, lambd)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class UpperBranch(nn.Module):
    def __init__(self, encoder_output_size):
        super(UpperBranch, self).__init__()
        self.conv_block = ConvBlock(encoder_output_size, encoder_output_size)
        self.grl_weight = 0.1

    def forward(self, features1, features2, grl_weight=0.1):

        features1_grl = grad_reverse(features1, grl_weight)
        features2_grl = grad_reverse(features2, grl_weight)

        conv_block_output1 = self.conv_block(features1_grl)
        conv_block_output2 = self.conv_block(features2_grl)

        conv_block_output1_flat = conv_block_output1.reshape(conv_block_output1.size(0), -1)
        conv_block_output2_flat = conv_block_output2.reshape(conv_block_output2.size(0), -1)

        cosine_similarity = F.cosine_similarity(conv_block_output1_flat, conv_block_output2_flat)
        f = 1 - cosine_similarity
        # f = torch.clamp(f, min=0.0, max=1.0)

        # Checking if values are within the range [0, 1]
        if torch.any(f < 0) or torch.any(f > 1):
            print("Rejecting sample due to out-of-range values:", f)
            return None, None, None
        
        # if torch.any(cosine_similarity < 0) or torch.any(cosine_similarity > 1):
        #     print("Rejecting sample due to out-of-range values:", cosine_similarity)
        #     return None, None, None

        # return cosine_similarity, conv_block_output1, conv_block_output2
        return f, conv_block_output1, conv_block_output2


class ClassificationBranch(nn.Module):
    def __init__(self, encoder_output_size, newsize, num_classes):
        super(ClassificationBranch, self).__init__()
        self.conv_block = ConvBlock(encoder_output_size, encoder_output_size)
        self.classifier = Classifier(newsize, num_classes)

    def forward(self, features):
        conv_block_output = self.conv_block(features)
        conv_block_output_flat = conv_block_output.reshape(conv_block_output.size(0), -1)
        classifier_output = self.classifier(conv_block_output_flat)
        return conv_block_output, classifier_output


# Decoder 3 ##########################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        def upsample_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResidualBlock(out_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            )
        
        self.up1 = upsample_block(in_channels, 256)
        self.up2 = upsample_block(256, 128)
        self.up3 = upsample_block(128, 64)
        self.up4 = upsample_block(64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()  # Assuming the original image has values in [0, 1]

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out_conv(x)
        x = self.out_activation(x)
        return x

class ReconstructionBranch(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ReconstructionBranch, self).__init__()
        self.decoder = Decoder(input_channels *2, output_channels)

    def forward(self, features1, features2):
        concatenated_features = torch.cat((features1, features2), dim=1)
        reconstruction = self.decoder(concatenated_features)
        return reconstruction

def grl_weight_scheduler(epoch, initial_weight=0.01, final_weight=1.0, step_size=10):
    steps = epoch // step_size
    increment = (final_weight - initial_weight) / ((final_weight / initial_weight) * step_size)
    weight = min(final_weight, initial_weight + steps * increment)
    return weight
