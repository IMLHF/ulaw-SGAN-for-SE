
from . import modules
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, var = {
      'DISCRIMINATOR_AD_MODEL': (modules.Module, modules.RealVariables),
  }[PARAM.model_name]

  return model_class, var
