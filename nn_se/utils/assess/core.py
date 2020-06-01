# -*- coding: utf-8 -*-
"""
@author: PengChuan
这一部分是语音信号的评价指标，用来评估语音信号降噪的质量，判断结果好坏
    pesq：perceptual evaluation of speech quality，语音质量听觉评估
    stoi：short time objective intelligibility，短时客观可懂度，尤其在低SNR下，可懂度尤其重要
    ssnr: segmental SNR，分段信噪比(时域指标)，它是参考信号和信号差的比值，衡量的是降噪程度
"""

import os
import tempfile
import numpy as np
import librosa
import platform
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources

from .. import audio

# import pesq binary
PESQ_PATH = os.path.split(os.path.realpath(__file__))[0]
if 'Linux' in platform.system():
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.ubuntu16.exe')
else:
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.win10.exe')


def calc_pesq(ref_sig, deg_sig, samplerate, is_file=False):

    if 'Windows' in platform.system():
        raise NotImplementedError

    if is_file:
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, ref_sig, deg_sig))
        msg = output.read()
    else:
        tmp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        tmp_deg = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        # librosa.output.write_wav(tmp_ref.name, ref_sig, samplerate)
        # librosa.output.write_wav(tmp_deg.name, deg_sig, samplerate)
        audio.write_audio(tmp_ref.name, ref_sig, samplerate)
        audio.write_audio(tmp_deg.name, deg_sig, samplerate)
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, tmp_ref.name, tmp_deg.name))
        msg = output.read()
        tmp_ref.close()
        tmp_deg.close()
        # os.unlink(tmp_ref.name)
        # os.unlink(tmp_deg.name)
    score = msg.split('Prediction : PESQ_MOS = ')
    # print(msg)
    # exit(0)
    # print(score)
    if len(score)<=1:
      print('calculate error.')
      return 2.0
    return float(score[1][:-1])


def calc_stoi(ref_sig, deg_sig, samplerate):
  return stoi(ref_sig, deg_sig, samplerate)


def calc_sdr(ref_sig, deg_sig, samplerate):
    """Calculate Source-to-Distortion Ratio(SDR).
    NOTE: one wav or batch wav.
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T] or [T], src_ref and src_deg must be same dimention.
        src_deg: numpy.ndarray, [C, T] or [T], reordered by best PIT permutation
    Returns:
        SDR
    """
    sdr, sir, sar, popt = bss_eval_sources(ref_sig, deg_sig)
    return sdr[0]


def extractOverlappedWindows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

ssnr_min = -10
ssnr_max = 35
# eps = np.finfo(np.float32).tiny
eps=np.finfo(np.float64).eps

def calc_SegSNR(ref_sig, deg_sig, frame_size, frame_shift):
    # ref_frame = librosa.util.frame(ref_sig, frame_length=frame_size, hop_length=frame_shift).T
    # deg_frame = librosa.util.frame(deg_sig, frame_length=frame_size, hop_length=frame_shift).T

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,frame_size+1)/(frame_size+1)))
    ref_frame=extractOverlappedWindows(ref_sig,frame_size,frame_size-frame_shift,hannWin)
    deg_frame=extractOverlappedWindows(deg_sig,frame_size,frame_size-frame_shift,hannWin)

    # print(np.shape(ref_frame), flush=True)
    noise_frame = ref_frame - deg_frame
    ref_energy = np.sum(ref_frame ** 2, axis=-1)
    noise_energy = np.sum(noise_frame ** 2, axis=-1) + eps
    ssnr = 10 * np.log10(ref_energy / noise_energy + eps)
    ssnr[ssnr < ssnr_min] = ssnr_min
    ssnr[ssnr > ssnr_max] = ssnr_max
    ssnr = ssnr[:-1]
    return np.mean(ssnr)
