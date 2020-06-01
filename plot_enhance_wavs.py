from nn_se import inference
from nn_se.utils import audio
from nn_se.utils import misc_utils
from nn_se import FLAGS
from matplotlib import pyplot as plt
import os
from pathlib import Path
import numpy as np
import librosa
import scipy
import tqdm
from nn_se.utils.assess.core import calc_pesq, calc_stoi, calc_sdr, calc_SegSNR
from nn_se import sepm

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

smg = None

noisy_and_ref_list = [
  # ('exp/paper_sample/0db_mix_ref/p265_015_nfree_0678.wav', 'exp/paper_sample/0db_mix_ref/p265_015.wav'),
  # ('exp/paper_sample/0db_mix_ref/p265_026_nfree_0571.wav', 'exp/paper_sample/0db_mix_ref/p265_026.wav'),
  # ('exp/paper_sample/0db_mix_ref/p265_038_nfree_0758.wav', 'exp/paper_sample/0db_mix_ref/p265_038.wav'),
  # ('exp/paper_sample/0db_mix_ref/p267_087_nfree_0663.wav', 'exp/paper_sample/0db_mix_ref/p267_087.wav'),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00002.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00003.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_utrA-v8pPm4_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10006_3RybHF5mX78_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10006_3RybHF5mX78_00002.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10006_5tGaUGO_z50_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10006_zQROl4ZsMVA_00002.wav', None),
  ('paper-sample/p232_036-noisy.wav', 'paper-sample/p232_036.wav'),
  ('paper-sample/p232_036-mdphd.wav', 'paper-sample/p232_036.wav'),
  ('paper-sample/p232_036-segan.wav', 'paper-sample/p232_036.wav'),
  ('paper-sample/dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna/40002/p232_036-ulawgan.wav',
   'paper-sample/p232_036.wav'),
  ('paper-sample/p232_170-noisy.wav', 'paper-sample/p232_170.wav'),
  ('paper-sample/p232_170-mdphd.wav', 'paper-sample/p232_170.wav'),
  ('paper-sample/p232_170-segan.wav', 'paper-sample/p232_170.wav'),
  ('paper-sample/dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna/40002/p232_170-ulawgan.wav',
   'paper-sample/p232_170.wav'),
  ('paper-sample/p232_415-noisy.wav', 'paper-sample/p232_415.wav'),
  ('paper-sample/p232_415-mdphd.wav', 'paper-sample/p232_415.wav'),
  ('paper-sample/p232_415-segan.wav', 'paper-sample/p232_415.wav'),
  ('paper-sample/dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna/40002/p232_415-ulawgan.wav',
   'paper-sample/p232_415.wav'),
  ('paper-sample/p257_070-noisy.wav', 'paper-sample/p257_070.wav'),
  ('paper-sample/p257_070-mdphd.wav', 'paper-sample/p257_070.wav'),
  ('paper-sample/p257_070-segan.wav', 'paper-sample/p257_070.wav'),
  ('paper-sample/dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna/40002/p257_070-ulawgan.wav',
   'paper-sample/p257_070.wav'),
  ('paper-sample/p257_395-noisy.wav', 'paper-sample/p257_395.wav'),
  ('paper-sample/p257_395-mdphd.wav', 'paper-sample/p257_395.wav'),
  ('paper-sample/p257_395-segan.wav', 'paper-sample/p257_395.wav'),
  ('paper-sample/dse_ulawV2_G_FTmagmse_Ndloss_ssnr_001_specAna/40002/p257_395-ulawgan.wav',
   'paper-sample/p257_395.wav'),

]
save_dir = 'exp/paper_sample'

def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  signal = np.array(signal, dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T

def enhance_and_calcMetrics(noisy_and_ref):
  # figsize = (2.8, 1.6)
  figsize = (6, 3.5)
  fontsize = 14
  noisy_wav_dir, ref_wav_dir = noisy_and_ref
  noisy_wav, sr = audio.read_audio(noisy_wav_dir)
  ref_wav, sr = audio.read_audio(ref_wav_dir) if ref_wav_dir else (None, sr)
  config_name = FLAGS.PARAM().config_name()
  noisy_stem = Path(noisy_wav_dir).stem
  ref_stem = Path(ref_wav_dir).stem
  enhanced_wav = noisy_wav

  ## plot enhanced_mag; save enhanced_wav; calc metrics [pesq]
  name_prefix = "%s_%s" % (config_name, noisy_stem)

  # calc metrics [pesq_noisy->pesq_enhanced | pesqi, stoi, sdr, ssnr]
  if ref_wav is not None:
    # pesq_enhanced = calc_pesq(ref_wav, enhanced_wav, sr)
    # stoi_enhanced = calc_stoi(ref_wav, enhanced_wav, sr)
    # sdr_enhanced = calc_sdr(ref_wav, enhanced_wav, sr)
    # ssnr_enhanced = calc_SegSNR(ref_wav, enhanced_wav, 480, 120)

    _, csigv, cbakv, cmosv, pesqv, ssnrv = sepm.compareone((ref_wav_dir, noisy_wav_dir))
    # from pypesq import pesq
    # print("")
    # print(pesq(ref_wav, enhanced_wav, sr))
    # print(ssnrv, pesqv)
    # print(ssnr_enhanced, pesq_enhanced)

    metrics_eval_ans_f = "%s_metrics_eval_ans.log" % config_name
    with open(os.path.join(save_dir, metrics_eval_ans_f), 'a') as f:
      f.write(name_prefix+":\n")
      f.write("    csig: %.4f, cbak: %.4f, cmos: %.4f, pesq: %.4f, ssnr: %.4f\n\n" % (
          csigv, cbakv, cmosv, pesqv, ssnrv))

  # get x_ticks
  enhanced_mag = magnitude_spectrum_librosa_stft(enhanced_wav, 512, 128*3)
  n_frame = np.shape(enhanced_mag)[0]
  n=0
  i=0
  x1 = []
  x2 = []
  while(n<n_frame):
    x1.append(n)
    x2.append(i)
    n = n+125
    i = i+1


  # enhanced_mag
  # enhanced_mag = magnitude_spectrum_librosa_stft(enhanced_wav, 256, 64*3)
  enhanced_mag = np.log(enhanced_mag+1e-1)
  plt.figure(figsize=figsize)
  # print(np.max(enhanced_mag), np.min(enhanced_mag))
  plt.pcolormesh(enhanced_mag.T, cmap='hot')
  plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
  # plt.title('STFT Magnitude')
  plt.xlabel('Time(S)', fontdict={'size':fontsize})
  plt.ylabel('Frequency(Hz)', fontdict={'size':fontsize})
  plt.xticks(x1, x2, size=fontsize)
  plt.yticks((0,66,128,194,257), ("0","2k","4k","6k","8k"), size=fontsize)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=fontsize)
  plt.savefig(os.path.join(save_dir,"%s.%s" % (name_prefix, "jpg")))
  # plt.show()
  plt.close()

  # enhanced_wav
  # audio.write_audio(os.path.join(save_dir, "%s_%s" % (name_prefix, "enhanced.wav")), enhanced_wav, sr)

  # noisy_mag
  # noisy_mag_file = os.path.join(save_dir, "%s_%s" % (noisy_stem, "noisy_mag.jpg"))
  # noisy_mag = magnitude_spectrum_librosa_stft(noisy_wav, 256, 64*3)
  # noisy_mag = np.log(noisy_mag*3+1e-2)
  # plt.figure(figsize=figsize)
  # plt.pcolormesh(noisy_mag.T, cmap='hot', vmin=-4.5, vmax=2.5)
  # plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
  # # plt.title('STFT Magnitude')
  # plt.xlabel('Time(S)')
  # plt.ylabel('Frequency(Hz)')
  # plt.xticks(x1, x2)
  # plt.yticks((0,66,128,194,257), ("0","2k","4k","6k","8k"))
  # plt.colorbar()
  # plt.savefig(noisy_mag_file)
  # # plt.show()
  # plt.close()

  # clean_mag
  clean_mag_file = os.path.join(save_dir, "%s.%s" % (ref_stem, "jpg"))
  if ref_wav is not None:
    clean_mag = magnitude_spectrum_librosa_stft(ref_wav, 512, 128*3)
    clean_mag = np.log(clean_mag+1e-1)
    plt.figure(figsize=figsize)
    plt.pcolormesh(clean_mag.T, cmap='hot')
    plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
    # plt.title('STFT Magnitude')

    plt.xlabel('Time(S)', fontdict={'size':fontsize})
    plt.ylabel('Frequency(Hz)', fontdict={'size':fontsize})
    plt.xticks(x1, x2, size=fontsize)
    plt.yticks((0,66,128,194,257), ("0","2k","4k","6k","8k"), size=fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)
    plt.savefig(clean_mag_file)
    # plt.show()
    plt.close()

if __name__ == "__main__":
  # enhance_and_calcMetrics(noisy_and_ref_list[0])
  for noisy_and_ref in tqdm.tqdm(noisy_and_ref_list, ncols=100):
    enhance_and_calcMetrics(noisy_and_ref)
