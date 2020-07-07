''' network architecture for Transformer v0 '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

BN_MOMENTUM = 0.1

class Transformer_v0(nn.Module):
    def __init__(self, nf=64, nframes=7, groups=7, front_RBs=5, back_RBs=10, L=8, Cp=128, CT=512, input_size=(64, 112), repeat_dynamic_times=6):
        super(Transformer_v0, self).__init__()
        self.nframes = nframes
        self.nf = nf
        self.center = nframes // 2
        self.L = L
        self.Cp = Cp
        self.CT = CT
        self.repeat_dynamic_times = repeat_dynamic_times
        self.input_size = input_size

        ResidualBlock_with_BN_f = functools.partial(arch_util.ResidualBlock_with_BN, nf=nf)

        #### extract features (for each frame)
        self.conv_first = nn.Sequential(
                nn.Conv2d(3, nf, 3, 1, 1, bias=True),
                nn.BatchNorm2d(nf, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
        )
        self.feature_extraction = arch_util.make_layer(ResidualBlock_with_BN_f, front_RBs)
        
        #### extract tokens from feature
        self.tokenizer = StaticTokenizer(nf, L, Cp, CT, input_size)        

        #### transform in token space
        self.transformer = StaticTransformer(CT)
        
        #### project tokens back to image sapce
        self.projector = StaticProjector(CT, nf)
        self.conv3x3_1 = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.BatchNorm2d(nf, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
        )

        #### Cascade DynamicTokenizer-DynamicTransformer-StaticProjector-Fusion
        dynamic_tokenizer_layers = []
        dynamic_transformer_layers = []
        static_projector_layers = []
        fusion_layers = []
        for i in range(repeat_dynamic_times):
            dynamic_tokenizer_layers.append(DynamicTokenizer(nf, L, Cp, CT, input_size))
            dynamic_transformer_layers.append(DynamicTransformer(CT, nframes * L))
            static_projector_layers.append(StaticProjector(CT, nf))
            fusion_layers.append(nn.Sequential(
                        nn.Conv2d(nf, nf, 3, 1, 1),
                        nn.BatchNorm2d(nf, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
            )
        self.dynamic_tokenizer_layers = nn.ModuleList(dynamic_tokenizer_layers)
        self.dynamic_transformer_layers = nn.ModuleList(dynamic_transformer_layers)
        self.static_projector_layers = nn.ModuleList(static_projector_layers)
        self.fusion_layers = nn.ModuleList(fusion_layers)

        #### reconstruction
        self.fea_fusion = nn.Conv2d(nframes*nf, nf, 3, 1, 1, bias=True)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_with_BN_f, back_RBs)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.last_conv = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.lrelu = nn.ReLU(inplace=False)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x_center = x[:, self.center, :, :, :].contiguous()
        feature = self.conv_first(x.view(-1, C, H, W))

        # feature extraction for each frame
        feature = self.feature_extraction(feature).view(B, N, -1, H, W) #(B, N, C(nf), H, W)

        # extract tokens from features
        tokens = self.tokenizer(feature)   #(BN, CT, L)

        # transform in token space
        group_tokens = []
        for i in range(self.nframes):
            group_tokens.append(self.transformer(tokens.view(B, N, self.CT, self.L), i))  #(B, CT, L)
        group_tokens = torch.stack(group_tokens, dim=1)   #(B, frames, CT, L)

        # project tokens back to image space
        project_features_first = self.projector(group_tokens.view(B*self.nframes, self.CT, self.L), feature[:, N//2, :, :, :].unsqueeze(1).repeat(1, self.nframes, 1, 1, 1).view(B*self.nframes, self.nf, H, W)).view(B, self.nframes, self.nf, H, W)   #(B, frames, nf, H, W)
        stage1_features = self.conv3x3_1(project_features_first.view(-1, self.nf, H, W)).view(B, self.nframes, self.nf, H, W)    #(B, frames, nf, H, W)

        # Cascade Tokenizer-Transformer-Projector
        previous_tokens = group_tokens  #(B, N, CT, L)
        previous_features = stage1_features #(B, N, nf, H ,W)
        for i in range(self.repeat_dynamic_times):
            stage2_tokens = self.dynamic_tokenizer_layers[i](previous_features, previous_tokens)    #(BN, CT, L)
            stage2_group_tokens = self.dynamic_transformer_layers[i](stage2_tokens.view(B, self.nframes, self.CT, self.L)).view(B, self.nframes, self.CT, self.L)    #(B, N, CT, L)
            stage2_project_features = self.static_projector_layers[i](stage2_group_tokens.view(-1, self.CT, self.L), previous_features.view(B*self.nframes, self.nf, H, W)).view(B, self.nframes, self.nf, H, W)  #(B, N, nf, H, W)
            stage2_features = self.fusion_layers[i](stage2_project_features.view(B*self.nframes, self.nf, H, W)).view(B, self.nframes, self.nf, H, W) #(B, N, nf, H ,w)

            # update
            previous_tokens = stage2_group_tokens   #(B, N, CT, L)
            previous_features = stage2_features     #(B, N, nf, H, W)

        # reconstruction
        fea = self.fea_fusion(previous_features.view(B, self.nframes*self.nf, H, W)) #(B, nf, H, W)
        out = self.recon_trunk(fea) #(B, nf, H ,W)

        # upsampling
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out))) #(B, nf, 2H, 2W)
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out))) #(B, 64, 4H, 4W)
        out = self.lrelu(self.HRconv(out))  #(B, 64, H, W)
        out = self.last_conv(out)   #(B, 3, H, W)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out = out+base
        return out


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
        token_coef = self.token_coef_conv(x.view(-1, C, H, W)).view(-1, self.L, H*W).permute(0, 2, 1)   # BN, HW, L
        token_coef = F.softmax(token_coef, dim=1)

        # get token_value
        value = self.value_conv(x.view(-1, C, H, W)).view(-1, C, H*W) # BN, C, HW
       
        # get tokens using token_coef and token_value
        tokens = torch.matmul(value, token_coef)  #BN, C, L

        # encode position from token_coef
        pos_encoding = self.pos_conv(token_coef, (H, W)) #(BN, Cp, L)
        
        # concat tokens and pos
        tokens = torch.cat([tokens, pos_encoding], dim=1)

        # fuse tokens and pos
        tokens = self.fuse_tokens_pos_conv(tokens)  #(BN, CT, L)
        
        return tokens


class DynamicTokenizer(nn.Module):
    def __init__(self, nf, L, Cp, CT, input_size):
        super(DynamicTokenizer, self).__init__()
        self.nf = nf
        self.L = L
        self.CT = CT
        self.q_conv = nn.Conv1d(CT, nf, 1, 1)
        self.k_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.v_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        #### Encode position from token_coef
        self.pos_conv = PosEncoder(input_size, Cp)

        #### Fuse tokens with position
        self.fuse_tokens_pos_conv = nn.Conv1d(nf+Cp, CT, 1, 1)

    def forward(self, feature, tokens):
        B, N, C, H, W = feature.shape
        feature = feature.view(-1, C, H, W)
        tokens = tokens.view(-1, self.CT, self.L)
        tokens_a, tokens_b = tokens[:, :, :self.L//2], tokens[:, :, self.L//2:]

        # compute token_coef using previous half tokens as query
        q = self.q_conv(tokens_a)   #(BN, C, L/2)
        k = self.k_conv(feature).view(-1, C, H*W).permute(0, 2, 1)  #(BN, HW, C)
        kq = torch.matmul(k, q) #(BN, HW, L/2)
        token_coef = F.softmax(kq, dim=1)   #(BN, HW, L/2)

        # get token_value
        v = self.v_conv(feature).view(-1, C, H*W)    #(BN, C, H*W)

        #get tokens using token_coef and token_value
        tokens = torch.matmul(v, token_coef)    #(BN, C, L/2)
        
        #encode position from token_coef
        pos_encoding = self.pos_conv(token_coef, (H, W))    #(BN, Cp, L/2)

        #concat tokens and pos
        tokens = torch.cat([tokens, pos_encoding], dim=1)   #(BN, C+Cp, L/2)

        #fuse tokens and pos
        tokens = self.fuse_tokens_pos_conv(tokens)  #(BN, CT, L/2)

        #concat tokens from previous
        tokens = torch.cat([tokens_b, tokens], dim=2)   #(BN, CT, L)

        return tokens


class PosEncoder(nn.Module):
    def __init__(self, desire_size, Cp):
        super(PosEncoder, self).__init__()
        self.desire_size = desire_size      #64, 112
        self.Cp = Cp
        self.pos_conv = nn.Conv1d(desire_size[0]*desire_size[1], Cp, 1, 1)

    def forward(self, token_coef, input_size):
        # token_coef (BN, HW, L), input_size(H, W)
        BN, HW, L = token_coef.size()
        fix_coef = token_coef.permute(0, 2, 1).contiguous().view(BN, L, input_size[0], input_size[1])   # (BN, L, H, W)
        fix_coef = F.interpolate(fix_coef, size=self.desire_size, mode='bilinear', align_corners=True)  # (BN, L, dH, dW)
        fix_coef = fix_coef.view(BN, L, -1).permute(0, 2, 1)    #(BN, dH*dW, L)
        out = self.pos_conv(fix_coef)   #(BN, Cp, L)
    
        return out


class StaticTransformer(nn.Module):
    def __init__(self, CT):   #total_tokens:  frames*L
        super(StaticTransformer, self).__init__()
        #### get attention of all frames, inter fusion
        self.center_conv = nn.Conv1d(CT, CT, 1, 1)
        self.nbr_conv = nn.Conv1d(CT, CT, 1, 1)
        self.inter_fusion_conv = nn.Conv1d(CT, CT, 1, 1)

        #### intra fusion
        self.k_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.q_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.v_conv = nn.Conv1d(CT, CT, 1, 1)
        
        #### last fusion
        self.last_fusion = nn.Conv1d(CT, CT, 1, 1)

    def forward(self, tokens, index):
        B, N, CT, L = tokens.shape
        center_frame_token = tokens[:, N//2, :, :]  #(B, CT, L)
        nbr_frame_token = tokens[:, index, :, :]    #(B, CT, L)
        
        # inter fusion
        center_frame_token = self.center_conv(center_frame_token)   #(B, CT, L)
        nbr_frame_token = self.nbr_conv(nbr_frame_token)    #(B, CT, L)
        cor = torch.matmul(center_frame_token.permute(0, 2, 1), nbr_frame_token)    #(B, L, L)
        cor = F.softmax(cor, dim = 2)   #(B, L, L)
        align = torch.matmul(nbr_frame_token, cor.permute(0, 2, 1)) #(B, CT, L)
        inter_fusion = align   #(B, CT, L)
        inter_fusion = self.inter_fusion_conv(inter_fusion) #(B, CT, L)

        # intra fusion
        k = self.k_conv(inter_fusion)   #(B, CT/2, L)
        q = self.q_conv(inter_fusion)   #(B, CT/2, L)
        v = self.v_conv(inter_fusion)   #(B, CT, L)
        kq = torch.matmul(k.permute(0, 2, 1), q)    #(B, L, L)
        kq = F.softmax(kq, dim=1)   #(B, L, L)
        kqv = torch.matmul(v, kq) + inter_fusion #(B, CT, L)
        token = self.last_fusion(kqv)    #(B, CT, L)

        return token


class DynamicTransformer(nn.Module):
    def __init__(self, CT, total_tokens):
        super(DynamicTransformer, self).__init__()

        #### get all attention of groups, inter fusion
        self.fuse_inter_conv = nn.Conv1d(CT, total_tokens, 1, 1)

        #### intra fusion
        self.k_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.q_conv = nn.Conv1d(CT, CT//2, 1, 1)
        self.v_conv = nn.Conv1d(CT, CT, 1, 1)

        #### last fusion
        self.last_fusion = nn.Conv1d(CT, CT, 1, 1)

    def forward(self, tokens):  #(B, N, CT, L)
        B, N, CT, L = tokens.shape

        # inter fusion
        inter_fusion = self.fuse_inter_conv(tokens.view(-1, CT, L)).view(B, N, -1, L)   #(B, N, NL, L)
        tokens_coef =  F.softmax(inter_fusion, dim=2)   #(B, N, NL, L)
        total_info = tokens.permute(0, 2, 1, 3).contiguous().view(B, CT, N*L)   #(B, CT, NL)
        inter = tokens + torch.matmul(total_info.unsqueeze(1).repeat(1, N, 1, 1), tokens_coef) #(B, N, CT, L)

        inter = inter.view(B*N, CT, L)  #(BN, CT, L)
        
        # intra fusion
        k = self.k_conv(inter)  #(BN CT/2, L)
        q = self.q_conv(inter)  #(BN, CT/2, L)
        v = self.v_conv(inter)  #(BN, CT, L)
        kq = torch.matmul(k.permute(0, 2, 1), q)    #(BN, L, L)
        kq = F.softmax(kq, dim=1)
        kqv = torch.matmul(v, kq) + inter   #(BN, CT, L)

        #last fusion
        fusion_tokens = self.last_fusion(kqv)    #(BN, CT, L)
        return fusion_tokens


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
