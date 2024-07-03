# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks

from transformers import PreTrainedModel, AutoModel

from .conv_stft import ConviSTFT, ConvSTFT
from .unet import UNet
from .frcrn_config import FRCRNConfig

import numpy as np
import math

BLOCK_MAPPING = {}

def stft_splitter(audio, n_fft=512, hop_len=128):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                hop_length=hop_len,
                                onesided=True,
                                return_complex=False)
        # print(audio_stft.size()) 
        return audio_stft[..., 0], audio_stft[..., 1] # 返回实部和虚部 B,F,T

def stft_mixer(real, imag, n_fft=512, hop_len=128):
    """
    real: B, F, T
    imag: B, F, T
    """
    # print(real.size())
    return torch.istft(
        torch.complex(real, imag),
        n_fft=n_fft, hop_length=hop_len, onesided=True
    )

class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        # padding = (kernel_size[0] // 2, 0)
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        # x = F.pad(x, [self.left_pad, 0])
        return self.conv(x)[..., :T]

class DPRnn(nn.Module):
    def __init__(self, input_ch, F_dim, hidden_ch):
        super(DPRnn, self).__init__()
        self.F_dim = F_dim
        self.input_size = input_ch
        self.hidden = hidden_ch
        self.intra_rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden // 2, bidirectional=True,
                                       batch_first=True)
        self.add_module('intra_rnn', self.intra_rnn)
        self.intra_fc = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.intra_ln = torch.nn.LayerNorm([F_dim, hidden_ch])

        self.inter_rnn = torch.nn.LSTM(input_size=self.hidden, hidden_size=self.hidden, batch_first=True)
        self.add_module('inter_rnn', self.inter_rnn)
        self.inter_fc = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.inter_ln = torch.nn.LayerNorm([F_dim, hidden_ch])

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        # czj
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        # czj
        
        B, C, F, T = x.size()

        x = x.permute(0, 3, 2, 1)  # B,T,F,C
        intra_in = torch.reshape(x, [B * T, F, C])
        intra_rnn_out, _ = self.intra_rnn(intra_in)
        intra_out = self.intra_ln(torch.reshape(self.intra_fc(intra_rnn_out), [B, T, F, C]))  # B,T,F,C
        intra_out = x + intra_out  # B,T,F,C

        inter_in = intra_out.permute(0, 2, 1, 3)  # B,F,T,C
        inter_in = torch.reshape(inter_in, [B * F, T, C])
        inter_rnn_out, _ = self.inter_rnn(inter_in)
        inter_out = self.inter_ln(
            torch.reshape(self.inter_fc(inter_rnn_out), [B, F, T, C]).permute(0, 2, 1, 3))  # B,T,F,C
        out = (intra_out + inter_out).permute(0, 3, 2, 1)
        return out

class IntegralAttention(nn.Module):
    def __init__(self, in_ch, u_path, n_head, freq_dim):
        super(IntegralAttention, self).__init__()
        self.in_ch = in_ch
        self.n_head = n_head
        self.freq_dim = freq_dim
        # 0.65R 79
        # 1R 59
        if u_path.find("nfft_1R.npy") != -1:
            temp = 59
        elif u_path.find("nfft.npy") != -1:
            temp = 79
        elif u_path.find("nfft_2R.npy") != -1:
            temp = 29
        if not os.path.exists(u_path):
            print(u_path)
            u_path = r"/Work21/2023/cuizhongjian/python/FrcrnWhisper/voicor_frcrn/U_640nfft_1R.npy"
            if not os.path.exists(u_path):
                u_path = r"/Work21/2023/cuizhongjian/python/FrcrnWhisper/voicor_frcrn/U_640nfft_1R.npy"
        self.register_buffer("u", torch.tensor([[np.load(u_path)[temp:, :].T]], dtype=torch.float)[:,:,:-1,:])  # 1x1x161x(672-79)
        # drawer.plot_mesh(self.u[0][0].data.T, "u")
        self.v_convs = nn.Sequential(
            nn.LayerNorm(freq_dim),
            nn.Conv2d(in_ch, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.k_convs = nn.Sequential(
            nn.LayerNorm(freq_dim),
            nn.Conv2d(in_ch, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.choosed_convs = nn.Sequential(
            nn.Conv2d(in_ch * n_head, self.in_ch * n_head, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.out_convs = nn.Sequential(
            nn.Conv2d(in_ch * n_head, self.in_ch, kernel_size=(1, 3), padding=(0, 1)),
        )

    def forward(self, x):
        # Harmonic integration
        v = self.v_convs(x)  # B,C*n_head,T,F -> V
        k = self.k_convs(x ** 2)  # B,C*n_head,T,F -> K
        atten = torch.matmul(k, self.u)  # B,C*n_head,T,candidates
        atten = F.softmax(atten, dim=-1) 
        H = torch.matmul(atten, self.u.permute(0, 1, 3, 2))
        choosed = self.choosed_convs(H)  # B,C*n_head,F,T
        v = choosed * v 
        return self.out_convs(v) 

class HarmonicAttention(nn.Module):
    def __init__(self, in_ch, out_ch, conv_ker, u_path, n_head, integral_atten=True, CFFusion=True, freq_dim=256):
        super(HarmonicAttention, self).__init__()

        self.conv_res = bool(in_ch == out_ch)
        self.out_ch = out_ch
        self.n_head = n_head
        self.integral_atten = integral_atten
        self.CFFusion = CFFusion
        self.in_ch = in_ch

        self.in_norm = nn.LayerNorm([in_ch, freq_dim])

        self.in_conv = nn.Sequential(
            CausalConv(in_ch, out_ch, kernel_size=conv_ker, stride=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

        if self.integral_atten:
            self.ln0 = nn.LayerNorm(freq_dim)
            self.integral_attention = IntegralAttention(in_ch=out_ch, u_path=u_path, n_head=n_head, freq_dim=freq_dim)

        if self.CFFusion:
            self.ln1 = nn.LayerNorm(freq_dim)
            self.channel_atten = MultiheadAttention(embed_dim=freq_dim, num_heads=8)

            self.ln2 = nn.LayerNorm(self.out_ch)
            self.f_atten = MultiheadAttention(embed_dim=self.out_ch, num_heads=2 if self.out_ch >= 8 else 1)

        self.dprnn = DPRnn(input_ch=out_ch, F_dim=freq_dim, hidden_ch=out_ch)
        # self.t_module = LSTM(in_dim=freq_dim * in_ch, hidden_ch=freq_dim, binary=False)

    def forward(self, s):
        """
        s: B,C,F,T
        """
        s = self.in_norm(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # BCFT->BTCF->BCFT

        if self.conv_res:
            s = self.in_conv(s) + s
        else:
            s = self.in_conv(s)
        B, C, F, T = s.size()
        s_ = s.permute(0, 1, 3, 2)  # B,C,T,F

        if self.integral_atten:
            ia = self.ln0(s_) 
            s_ = s_ + self.integral_attention(ia)  # B,C,T,F 

        if self.CFFusion:
            # channel attention
            ch_atten = self.ln1(s_).permute(1, 0, 2, 3).reshape(self.out_ch, -1, F)  # C,B*T,F
            ch_atten = self.channel_atten(ch_atten, ch_atten, ch_atten)[0]
            ch_atten = ch_atten.reshape(self.out_ch, B, T, F).permute(1, 0, 2, 3)
            s_ = s_ + ch_atten

            # frequency attention
            f_atten = self.ln2(s_.permute(3, 0, 2, 1).reshape(F, -1, self.out_ch))  # F,B*T,C
            f_atten = self.f_atten(f_atten, f_atten, f_atten)[0]
            f_atten = f_atten.reshape(F, B, T, self.out_ch).permute(1, 3, 2, 0)
            s_ = s_ + f_atten

        # temporal modeling
        # out = self.t_module(s_).permute(0, 1, 3, 2)  # BCTF->BCFT
        out = self.dprnn(s_.permute(0, 1, 3, 2))
        return out

class FRCRNDecorator(nn.Module):
    r""" A decorator of FRCRN for integrating into modelscope framework """
    config_class = FRCRNConfig

    def __init__(self, model_dir, complex, model_complexity, model_depth, log_amp, padding_mode, win_len, win_inc, fft_len, win_type, *args, **kwargs):
        """initialize the frcrn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__()
        model_dir = model_dir
        complex = complex
        model_complexity = model_complexity
        model_depth = model_depth
        log_amp = log_amp
        padding_mode = padding_mode
        win_len = win_len
        win_inc = win_inc
        fft_len = fft_len
        win_type = win_type
        self.model = FRCRN(complex, 
                           model_complexity, 
                           model_depth, 
                           log_amp, 
                           padding_mode, 
                           win_len,
                           win_inc,
                           fft_len,
                           win_type,
                           *args, 
                           **kwargs)
        model_bin_file = os.path.join(model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(
                model_bin_file, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # the new trained model by user is based on FRCRNDecorator
                self.load_state_dict(checkpoint['state_dict'])
            else:
                # The released model on Modelscope is based on FRCRN
                self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, inputs) -> Dict[str, Tensor]:
        result_list = self.model.forward(inputs)
        output = {
            'spec_l1': result_list[0],
            'wav_l1': result_list[1],
            'mask_l1': result_list[2],
            'spec_l2': result_list[3],
            'wav_l2': result_list[4],
            'mask_l2': result_list[5]
        }
        return output

class FRCRN(nn.Module):
    r""" Frequency Recurrent CRN """

    def __init__(self,
                 complex,
                 model_complexity,
                 model_depth,
                 log_amp,
                 padding_mode,
                 win_len=400,
                 win_inc=100,
                 fft_len=512,
                 win_type='hann',
                 **kwargs):
        r"""
        Args:
            complex: Whether to use complex networks.
            model_complexity: define the model complexity with the number of layers
            model_depth: Only two options are available : 10, 20
            log_amp: Whether to use log amplitude to estimate signals
            padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
            win_len: length of window used for defining one frame of sample points
            win_inc: length of window shifting (equivalent to hop_size)
            fft_len: number of Short Time Fourier Transform (STFT) points
            win_type: windowing type used in STFT, eg. 'hanning', 'hamming'
        """
        super().__init__()
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.istft = ConviSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.unet = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode)
        self.unet2 = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode)

    def forward(self, cmp_spec):
        out_list = []

        # [B, 2, D, T, 1]
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        # [B, 1, D, T, 2]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask1)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list

    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat([
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0]
            - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1],
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1]
            + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0]
        ], 1)
        # est_spec 对应 349 行的 refinement_out 和 350 行的 residual 
        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)

        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, noisy, labels, out_list, mode='Mix'):
        if mode == 'SiSNR':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 3:
                    loss = self.loss_1layer(noisy, est_spec, est_wav, labels,
                                            est_mask, mode)
            return dict(sisnr=loss)

        elif mode == 'Mix':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 3:
                    amp_loss, phase_loss, SiSNR_loss = self.loss_1layer(
                        noisy, est_spec, est_wav, labels, est_mask, mode)
                    loss = amp_loss + phase_loss + SiSNR_loss
            return dict(loss=loss, amp_loss=amp_loss, phase_loss=phase_loss)

    def loss_1layer(self, noisy, est, est_wav, labels, cmp_mask, mode='Mix'):
        r""" Compute the loss by mode
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        """
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            return -si_snr(est_wav, labels)
        elif mode == 'Mix':

            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            SiSNR_loss = -si_snr(est_wav, labels)

            b, d, t = est.size()
            S = self.stft(labels)
            Sr = S[:, :self.feat_dim, :]
            Si = S[:, self.feat_dim:, :]
            Y = self.stft(noisy)
            Yr = Y[:, :self.feat_dim, :]
            Yi = Y[:, self.feat_dim:, :]
            Y_pow = Yr**2 + Yi**2
            gth_mask = torch.cat([(Sr * Yr + Si * Yi) / (Y_pow + 1e-8),
                                  (Si * Yr - Sr * Yi) / (Y_pow + 1e-8)], 1)
            gth_mask[gth_mask > 2] = 1
            gth_mask[gth_mask < -2] = -1
            amp_loss = F.mse_loss(gth_mask[:, :self.feat_dim, :],
                                  cmp_mask[:, :self.feat_dim, :]) * d
            phase_loss = F.mse_loss(gth_mask[:, self.feat_dim:, :],
                                    cmp_mask[:, self.feat_dim:, :]) * d
            return amp_loss, phase_loss, SiSNR_loss

class Voicor_FRCRN(PreTrainedModel):
    config_class = FRCRNConfig
    def __init__(self, config: str, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        model_dir = config.model_dir
        complex = config.complex
        model_complexity = config.model_complexity
        model_depth = config.model_depth
        log_amp = config.log_amp
        padding_mode = config.padding_mode
        win_len = config.win_len
        win_type = config.win_type
        self.fft_len = config.fft_len
        self.hop_len = config.win_inc
        self.feat_dim = self.fft_len // 2 + 1
        
        n_head_num = 4 
        self.conv_ker = (5, 2)
        
        ''' Pre_enhancement module(PEM)'''
        self.frcrn=FRCRNDecorator(model_dir, complex, model_complexity, model_depth, log_amp, padding_mode, win_len, self.hop_len, self.fft_len, win_type, *args, **kwargs)
        
        '''Underlying information extractor'''
        self.extractor = nn.Sequential(
            HarmonicAttention(in_ch=2, out_ch=6, conv_ker=self.conv_ker, u_path=r"/Work21/2023/cuizhongjian/python/FrcrnWhisper/voicor_frcrn/U_640nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                              integral_atten=True, CFFusion=False),
            HarmonicAttention(in_ch=6, out_ch=12, conv_ker=self.conv_ker, u_path=r"/Work21/2023/cuizhongjian/python/FrcrnWhisper/voicor_frcrn/U_640nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                              integral_atten=True, CFFusion=False)
        )

        ''' Refinement'''
        self.refinement = nn.ModuleList()
        iter_num = 4  ## total number of refinement iterations
        self.iter_num = iter_num

        for i in range(iter_num):
            self.refinement.append(nn.Sequential(
                HarmonicAttention(in_ch=14, out_ch=6, conv_ker=self.conv_ker, u_path=r"/Work21/2023/cuizhongjian/python/FrcrnWhisper/voicor_frcrn/U_640nfft_1R.npy", n_head=n_head_num, freq_dim=self.fft_len//2,
                                  integral_atten=True, CFFusion=True),
                CausalConv(6, 2, kernel_size=self.conv_ker, stride=(1, 1))
            ))
            
    def forward(self, x):
        self.frcrn.eval()
        # [B, D*2, T]: B是batch，D是频域上面划分频段，T是时域的窗数
        with torch.no_grad():
            cmp_spec = self.frcrn.model.stft(x['noisy'])
        # [B, 1, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T] real_part/imag_part 将频域划分到时域和频域
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],
            cmp_spec[:, :, self.feat_dim:, :],
        ], 1)
        out = cmp_spec[:, :, :-1, :]
        
        ''' Pre_enhancement module(PEM)'''
        with torch.no_grad():
            frcrn_out = self.frcrn(cmp_spec)

        '''Underlying information extractor'''
        feature_head = self.extractor(out) # B,12,F,T
        
        '''multiple refinement iterator'''
        # [B, D*2, T]: B是batch，D是频域上面划分频段，T是时域的窗数
        refinement_out = frcrn_out['spec_l2']
        # [B, 1, D*2, T]
        refinement_out = torch.unsqueeze(refinement_out, 1)
        # to [B, 2, D, T] real_part/imag_part 将频域划分到时域和频域
        refinement_out = torch.cat([
            refinement_out[:, :, :self.feat_dim, :],
            refinement_out[:, :, self.feat_dim:, :],
        ], 1)[:, :, :-1, :]
        residual = refinement_out
        for idx in range(self.iter_num):
            feature_input = torch.cat((feature_head, residual),dim = 1)
            refinement = self.refinement[idx](feature_input)
            '''S-path'''
            residual = residual - refinement.detach()
            '''A-path'''
            refinement_out = refinement_out + refinement # B,2,F,T
        refinement_out = F.pad(refinement_out, [0, 0, 0, 1], value=1e-8)
        refinement_out = torch.cat([
            refinement_out[:, 0, :, :],
            refinement_out[:, 1, :, :],
        ], 1)
        refinement_out = self.frcrn.model.istft(refinement_out)
        refinement_out = torch.squeeze(refinement_out, 1)
        return refinement_out

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

AutoModel.register(FRCRNConfig, Voicor_FRCRN)