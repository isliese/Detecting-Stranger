# custom_layers.py
from keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)  # 'groups' 인수가 존재하면 제거
        return super().from_config(config)

# 커스텀 객체를 등록합니다.
from keras.utils import get_custom_objects
get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})
