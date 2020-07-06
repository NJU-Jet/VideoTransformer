''' network architecture for Transformer v0 '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class Transformer_v0(nn.Module):
    def __init__(self, nf=64, nframes=7, groups=64, front_RBs=5, back_RBs=10, L=8, Cp=128, CT=512, input_size=(64, 112)):
        super(Transformer_v0, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.L = L
        self.Cp = Cp
        self.CT = CT
        self.groups = groups
        self.input_size = input_size

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        
        #### extract tokens from feature
        self.tokenizer = StaticTokenizer(nf, L, Cp, CT, input_size)        

        #### transform in token space
        transformers = []
        for i in range(groups):
            transformers.append(
                    StaticTransformer(CT, nframes * L)
            )
        self.transformers = nn.ModuleList(transformers)

        #### project tokens back to image sapce
        self.projector = StaticProjector(CT, nf)
        self.conv3x3_1 = nn.Conv2d(nf, nf, 3, 1, 1)

        #### activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        B, N, C, H, W = x.size()
        x_center = x[:, self.center, :, :, :].contiguous()
        feature = self.lrelu(self.conv_first(x.view(-1, C, H, W)))

        #### feature extraction for each frame
        feature = self.feature_extraction(feature).view(B, N, -1, H, W) #(B, N, C, H, W)

        #### extract tokens from features
        tokens = self.tokenizer(feature)   #(BN, CT, L)

        #### transform in token space
        group_tokens = []
        for i in range(self.groups):
            group_tokens.append(self.transformers[i](tokens.view(B, N, self.CT, self.L)).unsqueeze(1))  #(B, 1, CT, L)
        group_tokens = torch.cat(group_tokens, dim=1)   #(B, groups, CT, L)

        #### project tokens back to image space
        project_features_first = self.projector(group_tokens.view(B*self.groups, self.CT, self.L), feature[:, N//2, :, :, :].repeat(self.groups, 1, 1, 1)).view(B, self.groups, self.nf, H, W)   #(B, groups, nf, H, W)
        state1_features = self.conv3x3_1(project_features_first.view(-1, self.nf, H, W)).view(B, self.groups, self.nf, H, W)    #(B, groups, nf, H, W)

        #### Cascade Tokenizer-Transformer-Projector


class StaticTokenizer(nn.Module):
    def __init__(self, C, L, Cp, CT, input_size):
        super(StaticTokenizer, self).__init__()
        self.C = C
        self.L = L
        self.input_size = input_size
        self.token_coef_conv = nn.Conv2d(C, L, 1, 1, 0)
        self.value_conv = nn.Conv2d(C, C, 1, 1, 0)

        #### Encode position from token_coef
        self.pos_conv = PosEncoder(input_size, Cp)

        #### Fuse tokens with pos
        self.fuse_tokens_pos_conv = nn.Conv1d(C+Cp, CT, 1, 1)


    def forward(self, x):
        B, N, C, H, W = x.size()
        # get token_coef
        token_coef = self.token_coef_conv(x.view(-1, C, H, W)).view(-1, self.L, H*W).unsqueeze(1).permute(0, 1, 3, 2)   # BN, 1, HW, L
        token_coef = F.softmax(token_coef, dim=2)

        # get token_value
        value = self.value_conv(x.view(-1, C, H, W)).view(-1, C, H*W).unsqueeze(1) # BN, 1, C, HW
       
        # get tokens using token_coef and token_value
        tokens = torch.matmul(value, token_coef).squeeze(1)  #BN, C, L

        # encode position from token_coef
        pos_encoding = self.pos_conv(token_coef, (H, W)) #(BN, Cp, L)
        
        # concat tokens and pos
        tokens = torch.cat([tokens, pos_encoding], dim=1)

        # fuse tokens and pos
        tokens = self.fuse_tokens_pos_conv(tokens)  #(BN, CT, L)
        
        return tokens


class PosEncoder(nn.Module):
    def __init__(self, desire_size, Cp):
        super(PosEncoder, self).__init__()
        self.desire_size = desire_size      #64, 112
        self.Cp = Cp
        self.pos_conv = nn.Conv1d(desire_size[0]*desire_size[1], Cp, 1, 1)

    def forward(self, token_coef, input_size):
        # token_coef (BN, 1, HW, L), input_size(H, W)
        BN, _, HW, L = token_coef.size()
        fix_coef = token_coef.squeeze(1).permute(0, 2, 1).contiguous().view(BN, L, input_size[0], input_size[1])   # (BN, L, H, W)
        fix_coef = F.interpolate(fix_coef, size=self.desire_size, mode='bilinear', align_corners=True)  # (BN, L, dH, dW)
        fix_coef = fix_coef.view(BN, L, -1).permute(0, 2, 1)    #(BN, dH*dW, L)
        out = self.pos_conv(fix_coef)   #(BN, Cp, L)
    
        return out

class StaticTransformer(nn.Module):
    def __init__(self, CT, total_tokens):   #total_tokens:  frames*L
        super(StaticTransformer, self).__init__()
        # get attention of all frames, inter fusion
        self.frame_att_conv = nn.Conv1d(CT, total_tokens, 1, 1)

        # intra fusion
        self.k_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.q_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.v_conv = nn.Conv1d(CT, CT, 1, 1)
        
        # last fusion
        self.last_fusion = nn.Conv1d(CT, CT, 1, 1)

    def forward(self, tokens):
        B, N, CT, L = tokens.shape
        center_frame_token = tokens[:, N//2, :, :]  #(B, CT, L)

        # inter fusion
        total_info = tokens.permute(0, 2, 1, 3).contiguous().view(B, CT, N*L)   #(B, CT, NL)
        tokens_coef = self.frame_att_conv(center_frame_token)   #(B, NL, L)
        tokens_coef = F.softmax(tokens_coef, dim=1) #(B, NL, L)
        inter_fusion = center_frame_token + torch.matmul(total_info, tokens_coef) #(B, CT, L)
    
        # intra fusion
        k = self.k_conv(inter_fusion)   #(B, CT/2, L)
        q = self.q_conv(inter_fusion)   #(B, CT/2, L)
        v = self.v_conv(inter_fusion)   #(B, CT, L)
        kq = torch.matmul(k.permute(0, 2, 1), q)    #(B, L, L)
        kq = F.softmax(kq, dim=1)   #(B, L, L)
        kqv = torch.matmul(v, kq) + center_frame_token  #(B, CT, L)
        token = center_frame_token + self.last_fusion(kqv)    #(B, CT, L)

        return token


class StaticProjector(nn.Module):
    def __init__(self, CT, C):
        super(StaticProjector, self).__init__()
        self.CT = CT
        self.C = C
        self.k_conv = nn.Conv1d(CT, C, 1, 1)
        self.q_conv = nn.Conv2d(C, C, 1, 1)
        self.v_conv = nn.Conv1d(CT, C, 1, 1)

    def forward(self, tokens, feature):
        B, C, H, W = feature.shape
        k = self.k_conv(tokens) #(B, C, L)
        q = self.q_conv(feature).view(B, C, -1).permute(0, 2, 1)    #(B, HW, C)
        kq = torch.matmul(q, k) #(B, HW, L)
        kq = F.softmax(kq, dim=2)   #(B, HW, L)
        v = self.v_conv(tokens) #(B, C, L)
        kqv = torch.matmul(v, kq.permute(0, 2, 1)).view(B, C, H, W) #(B, C, H, W)
        
        return feature + kqv
