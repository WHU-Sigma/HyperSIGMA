# -*- coding: utf-8 -*-

#from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor_vit import LayerDecayOptimizerConstructor_ViT
# from .layer_decay_optimizer_constructor_vitae import LayerDecayOptimizerConstructor_ViTAE
from .custom_layer_decay_optimizer_constructor import CustomLayerDecayOptimizerConstructor_InternImage
# from .layer_decay_optimizer_constructor_vitadapter import LayerDecayOptimizerConstructor_ViTadapter
#
# __all__ = ['LayerDecayOptimizerConstructor_ViT',
#            'LayerDecayOptimizerConstructor_ViTAE',
#            'LayerDecayOptimizerConstructor_ViTadapter',
#            'CustomLayerDecayOptimizerConstructor_InternImage']

__all__ = ['LayerDecayOptimizerConstructor_ViT',
           'CustomLayerDecayOptimizerConstructor_InternImage']
