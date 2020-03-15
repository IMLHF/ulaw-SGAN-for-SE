import tensorflow as tf
import os
import numpy as np
import time
import collections
from pathlib import Path
import sys


from .models import model_builder
from .models import modules
from .dataloader import dataloader
from .utils import misc_utils
from .FLAGS import PARAM

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("avg_sum_loss_G", "avg_sum_loss_D", "avg_show_losses", "cost_time"))):
  pass


def eval_one_epoch(sess, val_model, initializer):
  sess.run(initializer)
  val_s_time = time.time()
  ont_batch_time = time.time()

  i = 0
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  avg_sum_loss_G = None
  avg_sum_loss_D = None
  avg_show_losses = None
  while True:
    try:
      (sum_loss_G, sum_loss_D, show_losses,
       ) = sess.run([val_model.losses.sum_loss_G,
                     val_model.losses.sum_loss_D,
                     val_model.losses.show_losses,
                     ])
      if avg_sum_loss_G is None:
        avg_sum_loss_G = sum_loss_G
        avg_sum_loss_D = sum_loss_D
        avg_show_losses = show_losses
      else:
        avg_sum_loss_G += sum_loss_G
        avg_sum_loss_D += sum_loss_D
        avg_show_losses += show_losses
      i += 1
      print("\r", end="")
      print("validate: %d/%d, cost %.2fs, sum_loss[G %.2f, D %.2f], show_losses %s"
            "          " % (
                i, total_i, time.time()-ont_batch_time, sum_loss_G, sum_loss_D,
                str(show_losses)
            ),
            flush=True, end="")
      ont_batch_time = time.time()
    except tf.errors.OutOfRangeError:
      break

  print("\r", end="")
  avg_sum_loss_G /= total_i
  avg_sum_loss_D /= total_i
  avg_show_losses /= total_i
  return EvalOutputs(avg_sum_loss_G=avg_sum_loss_G,
                     avg_sum_loss_D=avg_sum_loss_D,
                     avg_show_losses=avg_show_losses,
                     cost_time=time.time()-val_s_time)


def main():
  train_log_file = misc_utils.train_log_file_dir()
  ckpt_dir = misc_utils.ckpt_dir()
  hparam_file = misc_utils.hparams_file_dir()
  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  if not ckpt_dir.exists():
    os.mkdir(str(ckpt_dir))

  misc_utils.save_hparams(str(hparam_file))

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      noisy_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_noisy_set)
      clean_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_clean_set)
      noisy_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_noisy_set)
      clean_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_clean_set)
      train_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_trainset_wav,
                                                                        clean_trainset_wav)
      val_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_valset_wav,
                                                                      clean_valset_wav)

    ModelC, VariablesC = model_builder.get_model_class_and_var()

    variables = VariablesC()
    train_model = ModelC(PARAM.MODEL_TRAIN_KEY, variables, train_inputs.mixed, train_inputs.clean)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(PARAM.MODEL_VALIDATE_KEY, variables, val_inputs.mixed,val_inputs.clean)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    # misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  # config.gpu_options.per_process_gpu_memory_fraction = PARAM.GPU_PARTION
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  # region validation before training
  misc_utils.print_log("\n\n", train_log_file)
  misc_utils.print_log("sum_losses_G: "+str(PARAM.sum_losses_G)+"\n", train_log_file)
  misc_utils.print_log("sum_losses_D: "+str(PARAM.sum_losses_D)+"\n", train_log_file)
  misc_utils.print_log("show losses: "+str(PARAM.show_losses)+"\n", train_log_file)
  evalOutputs_prev = eval_one_epoch(sess, val_model, val_inputs.initializer)
  misc_utils.print_log("                                            "
                       "                                            "
                       "                                         \n",
                       train_log_file, no_time=True)
  val_msg = "PRERUN.val> sum_loss:[G %.4F, D %.4f], show_losses:%s, Time:%.2Fs.\n" % (
      evalOutputs_prev.avg_sum_loss_G,
      evalOutputs_prev.avg_sum_loss_D,
      evalOutputs_prev.avg_show_losses,
      evalOutputs_prev.cost_time)
  misc_utils.print_log(val_msg, train_log_file)
  # endregion

  avg_sum_loss_G = None
  avg_sum_loss_D = None
  avg_show_losses = None
  save_time = time.time()
  sess.run(train_inputs.initializer)
  while True:
    try:
      one_batch_time = time.time()
      (sum_loss_G, sum_loss_D, show_losses, _,
       global_step, lr, u,
       ) = sess.run([train_model.losses.sum_loss_G,
                     train_model.losses.sum_loss_D,
                     train_model.losses.show_losses,
                     train_model.train_op,
                     train_model.global_step,
                     train_model.lr,
                     train_model.variables._f_u,
                     ])
      if global_step > PARAM.max_step:
        sess.close()
        misc_utils.print_log("\n", train_log_file, no_time=True)
        msg = '################### Training Done. ###################\n'
        misc_utils.print_log(msg, train_log_file)
        break
      if avg_sum_loss_G is None:
        avg_sum_loss_G = sum_loss_G
        avg_sum_loss_D = sum_loss_D
        avg_show_losses = show_losses
      else:
        avg_sum_loss_G += sum_loss_G
        avg_sum_loss_D += sum_loss_D
        avg_show_losses += show_losses

      print("\r", end="")
      u_str = "#(u %.2e)" % u if len(PARAM.sum_losses_D)>0 else "          "
      print("train step: %d/%d, cost %.2fs, sum_loss[G %.2f, D %.2f], show_losses %s %s          " % (
            global_step, PARAM.max_step, time.time()-one_batch_time, sum_loss_G, sum_loss_D,
            str(show_losses), u_str),
            flush=True, end="")
      one_batch_time = time.time()

      if global_step % PARAM.step_to_save == 0:
        print("\r", end="")
        avg_sum_loss_G /= PARAM.step_to_save
        avg_sum_loss_D /= PARAM.step_to_save
        avg_show_losses /= PARAM.step_to_save
        misc_utils.print_log("                                                                        "
                             "                                                                        "
                             "                             \n\n", train_log_file, no_time=True)
        misc_utils.print_log("  Save step %03d:\n" % global_step, train_log_file)
        misc_utils.print_log("     sum_losses_G: "+str(PARAM.sum_losses_G)+"\n", train_log_file)
        misc_utils.print_log("     sum_losses_D: "+str(PARAM.sum_losses_D)+"\n", train_log_file)
        misc_utils.print_log("     show losses : "+str(PARAM.show_losses)+"\n", train_log_file)
        misc_utils.print_log("     Train     > sum_loss:[G %.4f, D %.4f], show_losses:%s, %s, Time:%ds.      \n" % (
            avg_sum_loss_G, avg_sum_loss_D, str(avg_show_losses), u_str, time.time()-save_time),
            train_log_file)

        # val
        evalOutputs = eval_one_epoch(sess, val_model, val_inputs.initializer)
        misc_utils.print_log("     Validation> sum_loss:[G %.4f, D %.4f], show_losses:%s, Time:%ds.            \n" % (
            evalOutputs.avg_sum_loss_G, evalOutputs.avg_sum_loss_D,
            str(evalOutputs.avg_show_losses), evalOutputs.cost_time),
            train_log_file)

        # save ckpt
        ckpt_name = PARAM().config_name()+('_step%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
            global_step, avg_sum_loss_G, evalOutputs.avg_sum_loss_G, lr,
            time.time()-save_time))
        train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))

        # init var
        avg_sum_loss_G = None
        avg_sum_loss_D = None
        avg_show_losses = None
        save_time = time.time()
    except tf.errors.OutOfRangeError:
      sess.run(train_inputs.initializer)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
