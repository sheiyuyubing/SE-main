import os
import torch
from torch.utils.data import Dataset
import torchaudio
from functions import *

class GetDataset(Dataset):
    def __init__(self, method):
        self.method = method
        # 读取训练集或测试集中的文件名
        with open('C:\\Users\\31986\\Desktop\\SE-main\\dataset\\test.txt', 'r') as f:
            self.file_names = [line.strip()[-12:] for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        try:
            file_name = self.file_names[idx]

            # 设置文件路径
            clean_path = os.path.join('C:\\Users\\31986\\Desktop\\SE-main\\dataset\\wavs_clean', file_name)
            noisy_path = os.path.join('C:\\Users\\31986\\Desktop\\SE-main\\dataset\\wavs_noisy', file_name)

            # 加载干净和带噪的音频文件
            clean_waveform, _ = torchaudio.load(clean_path)
            noisy_waveform, _ = torchaudio.load(noisy_path)

            clean_waveform = clean_waveform[0]  # 假设是单通道音频
            noisy_waveform = noisy_waveform[0]
            noise_waveform = noisy_waveform - clean_waveform

            # 计算STFT
            clean_stft = calculate_stft(clean_waveform)
            noisy_stft = calculate_stft(noisy_waveform)
            noise_stft = calculate_stft(noise_waveform)

            # 生成mask
            if self.method == 'iam':
                mask = calculate_iam(clean_stft, noisy_stft)
            elif self.method == 'psm':
                mask = calculate_psm(clean_stft, noisy_stft)
            elif self.method == 'irm':
                mask = calculate_irm(clean_stft, noise_stft)
            elif self.method == 'orm':
                mask = calculate_orm(clean_stft, noise_stft)
            elif self.method == 'ibm':
                mask = calculate_ibm(clean_stft, noise_stft)
            else:  # 默认使用干净的STFT作为mask
                mask = torch.abs(clean_stft)

            feature = torch.unsqueeze(torch.abs(noisy_stft).T, 0)  # feature维度：[channel=1, time_frames, freq_bins]
            label = torch.unsqueeze(mask.T, 0)  # label维度同feature

            return {'feature': feature, 'label': label}

        except Exception as e:
            print(f"加载第{idx}个样本时出错: {e}")
            return None  # 如果出错，返回None，避免中断训练
