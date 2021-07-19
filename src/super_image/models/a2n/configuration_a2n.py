from ...configuration_utils import PretrainedConfig


class A2nConfig(PretrainedConfig):
    model_type = 'A2N'

    def __init__(self, scale=None, supported_scales=None, data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.supported_scales = supported_scales
        self.data_parallel = data_parallel

        if scale is not None:
            if supported_scales is None:
                raise ValueError(f'Input supported_scales is not defined. You need to define a list of '
                                 f'supported_scales that includes your scale, e.g. [{scale}]')
            else:
                if scale not in supported_scales:
                    raise ValueError(f'Input scale {scale} is not in supported scales for this '
                                     f'model: {supported_scales}')
