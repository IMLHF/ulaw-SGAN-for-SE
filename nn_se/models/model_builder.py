
from . import modules
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, g, d = {
      'DISCRIMINATOR_AD_MODEL': (modules.Module, modules.Generator, modules.Discriminator),
  }[PARAM.model_name]

  return model_class, g, d
