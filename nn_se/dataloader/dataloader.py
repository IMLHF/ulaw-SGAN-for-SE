import tensorflow as tf
import collections
import os
from pathlib import Path

from ..FLAGS import PARAM
from ..utils import misc_utils
from ..utils import audio


class DataSetsOutputs(
    collections.namedtuple("DataSetOutputs",
                           ("initializer", "clean", "mixed"))):
  pass


# def parse_func(record_proto):
#   wav_len = int(PARAM.sampling_rate*PARAM.train_val_wav_seconds)
#   features = {
#       'clean': tf.io.FixedLenFeature([wav_len], tf.float32),
#       'mixed': tf.io.FixedLenFeature([wav_len], tf.float32)
#   }
#   record = tf.io.parse_single_example(record_proto, features=features)
#   return record['clean'], record['mixed']


# def get_batch_inputs_from_dataset(sub_dataset_name, shuffle_records=True):
#   """
#   dataset_name: PARAM.train_name, PARAM.val_name, PARAM.test_name
#   """
#   tfrecords_list = misc_utils.datasets_dir().joinpath(sub_dataset_name, "tfrecords", "*.tfrecords")
#   files = tf.data.Dataset.list_files(str(tfrecords_list))
#   # files = files.take(FLAGS.PARAM.MAX_TFRECORD_FILES_USED)
#   if shuffle_records:
#     files = files.shuffle(PARAM.tfrecords_num_pre_set)
#   if not shuffle_records:
#     dataset = files.interleave(tf.data.TFRecordDataset,
#                                cycle_length=1,
#                                block_length=PARAM.batch_size,
#                                # num_parallel_calls=1,
#                                )
#   else:  # shuffle
#     dataset = files.interleave(tf.data.TFRecordDataset,
#                                cycle_length=8,
#                                block_length=PARAM.batch_size//8,
#                                num_parallel_calls=PARAM.n_processor_tfdata,
#                                )
#   if shuffle_records:
#     dataset = dataset.shuffle(PARAM.batch_size*10)

#   dataset = dataset.map(parse_func, num_parallel_calls=PARAM.n_processor_tfdata)
#   dataset = dataset.batch(batch_size=PARAM.batch_size, drop_remainder=True)
#   # dataset = dataset.prefetch(buffer_size=PARAM.batch_size)
#   dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
#   clean, mixed = dataset_iter.get_next()
#   return DataSetsOutputs(initializer=dataset_iter.initializer,
#                          clean=clean,
#                          mixed=mixed)


def _generator(noisy_list, clean_list):
  """
  return clean, noisy
  """
  for noisy_dir, clean_dir in zip(noisy_list, clean_list):
    noisy, nsr = audio.read_audio(noisy_dir)
    clean, csr = audio.read_audio(clean_dir)
    assert nsr == csr, "sample rate error."
    wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
    # clean = audio.repeat_to_len(clean, wav_len)
    # noisy = audio.repeat_to_len(noisy, wav_len)
    clean, noisy = audio.repeat_to_len_2(clean, noisy, wav_len, True)
    yield clean, noisy


def get_batch_inputs_from_nosiyCleanDataset(noisy_path, clean_path, shuffle_records=True):
  """
  noisy_path: noisy wavs path
  clean_path: clean wavs path
  """
  noisy_path = Path(noisy_path)
  clean_path = Path(clean_path)
  noisy_list = list(map(str, noisy_path.glob("*.wav")))
  clean_list = list(map(str, clean_path.glob("*.wav")))
  noisy_list.sort()
  clean_list.sort()

  wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
  dataset = tf.data.Dataset.from_generator(_generator, output_types=(tf.float32,tf.float32),
                                           output_shapes=(
                                               tf.TensorShape([wav_len]),
                                               tf.TensorShape([wav_len])),
                                           args=(noisy_list, clean_list))

  if shuffle_records:
    dataset = dataset.shuffle(PARAM.batch_size*10)

  dataset = dataset.batch(batch_size=PARAM.batch_size, drop_remainder=True)
  # dataset = dataset.prefetch(buffer_size=PARAM.batch_size)
  dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
  clean, mixed = dataset_iter.get_next()
  return DataSetsOutputs(initializer=dataset_iter.initializer,
                         clean=clean,
                         mixed=mixed)
