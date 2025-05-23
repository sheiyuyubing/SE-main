import sys
import os
import torchaudio
import torch
import soundfile as sf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit
import PyQt5.QtCore
import numpy as np


# 假设已有的语音增强方法
def enhance_audio(input_path, output_path, method='ibm'):
    # 加载带噪音的音频
    noisy_waveform, sample_rate = torchaudio.load(input_path)

    # 语音增强方法（根据选择的方式进行处理）
    if method == 'ibm':
        enhanced_waveform = ibm_enhancement(noisy_waveform)
    elif method == 'irm':
        enhanced_waveform = irm_enhancement(noisy_waveform)
    elif method == 'iam':
        enhanced_waveform = iam_enhancement(noisy_waveform)
    elif method == 'psm':
        enhanced_waveform = psm_enhancement(noisy_waveform)
    elif method == 'orm':
        enhanced_waveform = orm_enhancement(noisy_waveform)

    # 保存增强后的音频
    sf.write(output_path, enhanced_waveform.numpy().T, sample_rate)
    print(f"增强后的音频已保存到: {output_path}")


# 示例的增强方法（这些方法可以根据具体实现来修改）
def ibm_enhancement(noisy_waveform):
    return noisy_waveform * 0.8  # 模拟一个增强效果


def irm_enhancement(noisy_waveform):
    return noisy_waveform * 0.7


def iam_enhancement(noisy_waveform):
    return noisy_waveform * 0.9


def psm_enhancement(noisy_waveform):
    return noisy_waveform * 0.6


def orm_enhancement(noisy_waveform):
    return noisy_waveform * 0.85


class SpeechEnhancementUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('语音增强界面')
        self.setGeometry(300, 300, 500, 400)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 文件选择控件
        self.noisy_audio_label = QLabel('选择带噪音的音频文件：')
        self.noisy_audio_input = QLineEdit(self)
        self.noisy_audio_input.setPlaceholderText('选择带噪音的音频文件')
        self.noisy_audio_btn = QPushButton('选择文件', self)
        self.noisy_audio_btn.clicked.connect(self.select_noisy_audio)

        self.enhanced_audio_label = QLabel('选择保存增强后音频文件：')
        self.enhanced_audio_input = QLineEdit(self)
        self.enhanced_audio_input.setPlaceholderText('选择保存增强后的音频文件')
        self.enhanced_audio_btn = QPushButton('选择保存路径', self)
        self.enhanced_audio_btn.clicked.connect(self.select_enhanced_audio_path)

        # 选择增强方法
        self.method_label = QLabel('选择语音增强方法：')
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(['ibm', 'irm', 'iam', 'psm', 'orm'])

        # 执行按钮
        self.enhance_btn = QPushButton('开始增强', self)
        self.enhance_btn.clicked.connect(self.start_enhancement)

        # 将控件添加到布局中
        layout.addWidget(self.noisy_audio_label)
        layout.addWidget(self.noisy_audio_input)
        layout.addWidget(self.noisy_audio_btn)
        layout.addWidget(self.enhanced_audio_label)
        layout.addWidget(self.enhanced_audio_input)
        layout.addWidget(self.enhanced_audio_btn)
        layout.addWidget(self.method_label)
        layout.addWidget(self.method_combo)
        layout.addWidget(self.enhance_btn)

        self.setLayout(layout)

    def select_noisy_audio(self):
        file, _ = QFileDialog.getOpenFileName(self, '选择带噪音的音频文件', '', 'Audio Files (*.wav)')
        if file:
            self.noisy_audio_input.setText(file)

    def select_enhanced_audio_path(self):
        file, _ = QFileDialog.getSaveFileName(self, '选择保存路径', '', 'Audio Files (*.wav)')
        if file:
            self.enhanced_audio_input.setText(file)

    def start_enhancement(self):
        noisy_audio_path = self.noisy_audio_input.text()
        enhanced_audio_path = self.enhanced_audio_input.text()
        selected_method = self.method_combo.currentText()

        if not noisy_audio_path or not enhanced_audio_path:
            print("请确保选择了输入文件和保存路径")
            return

        # 执行增强操作
        enhance_audio(noisy_audio_path, enhanced_audio_path, method=selected_method)
        print("增强完成，文件已保存")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpeechEnhancementUI()
    ex.show()
    sys.exit(app.exec_())