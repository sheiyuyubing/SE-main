import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义文件类型
file_types = ['clean', 'noisy', 'ibm', 'iam', 'irm', 'psm', 'orm']
fs = 16000  # 采样率

# 设置音频文件所在的绝对路径
audio_dir = r'C:\Users\31986\Desktop\SE-main\p257_028'  # 请根据实际情况修改为正确的路径

for file_type in file_types:
    # 使用绝对路径拼接音频文件路径
    path = os.path.join(audio_dir, f'{file_type}.wav')
    print(f"加载文件路径：{path}")

    try:
        # 加载音频文件
        wave, _ = librosa.load(path, sr=fs)
    except FileNotFoundError:
        print(f"文件 {path} 未找到！")
        continue
    except Exception as e:
        print(f"加载文件 {path} 时出错：{e}")
        continue

    # 归一化波形
    wave = wave * 1.0 / (max(abs(wave)))

    # 创建图像
    plt.figure(figsize=(12, 6))

    # 绘制频谱图
    plt.subplot(1, 2, 1)
    D = np.abs(librosa.stft(wave))
    plt.imshow(librosa.power_to_db(D ** 2, ref=np.max), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{file_type} - 语谱图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')

    # 绘制波形图
    plt.subplot(1, 2, 2)
    times = np.arange(len(wave)) / float(fs)
    plt.plot(times, wave)
    plt.title(f'{file_type} - 波形图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

