import torch
import torch.nn as nn

'''
    < ConvBlock >
    Small unit block consists of [convolution layer - normalization layer - non linearity layer]
    
    * Parameters
    1. in_dim : Input dimension(channels number)
    2. out_dim : Output dimension(channels number)
    3. k : Kernel size(filter size)
    4. s : stride
    5. p : padding size
    6. norm : If it is true add Instance Normalization layer, otherwise skip this layer
    7. non_linear : You can choose between 'leaky_relu', 'relu', 'None'
'''
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []
        
        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
            
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        
        self.conv_block = nn.Sequential(* layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
    

'''
    < Discriminator >
    
    PatchGAN discriminator. See https://arxiv.org/pdf/1611.07004 6.1.2 Discriminator architectures.
    It uses two discriminator which have different output sizes(different local probabilities).
    
    Futhermore, it is conditional discriminator so input dimension is 6. You can make input by concatenating
    two images to make pair of Domain A image and Domain B image. 
    There are two cases to concatenate, [Domain_A, Domain_B_ground_truth] and [Domain_A, Domain_B_generated]
    
    d_1 : (N, 6, 128, 128) -> (N, 1, 14, 14)
    d_2 : (N, 6, 128, 128) -> (N, 1, 30, 30)
    
    In training, the generator needs to fool both of d_1 and d_2 and it makes the generator more robust.
 
'''  
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()       
        # Discriminator with last patch (14x14)
        # (N, 3, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 ConvBlock(3, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))
        
        # Discriminator with last patch (30x30)
        # (N, 3, 128, 128) -> (N, 1, 30, 30)
        self.d_2 = nn.Sequential(ConvBlock(3, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 256, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(256, 1, k=4, s=1, p=1, norm=False, non_linear=None))
    
    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)
    
        
        
        
        
        
if __name__ == "__main__":
    Discriminator = Discriminator().cuda()
    Discriminator.eval()
    
    image = torch.randn(16, 3, 128, 128).cuda()
    
    with torch.no_grad():
        BOO = Discriminator(image)
        print(BOO[0].shape, BOO[1].shape)
        
        
        
        