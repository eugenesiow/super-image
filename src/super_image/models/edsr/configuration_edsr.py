from ...configuration_utils import PretrainedConfig


class EdsrConfig(PretrainedConfig):
    model_type = 'EDSR'

    def __init__(self, scale=None, supported_scales=None, n_resblocks=16, n_feats=64, n_colors=3, rgb_range=255,
                 res_scale=1, data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.supported_scales = supported_scales
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.data_parallel = data_parallel

        if scale is not None:
            if supported_scales is None:
                raise ValueError(f'Input supported_scales is not defined. You need to define a list of '
                                 f'supported_scales that includes your scale, e.g. [{scale}]')
            else:
                if scale not in supported_scales:
                    raise ValueError(f'Input scale {scale} is not in supported scales for this '
                                     f'model: {supported_scales}')
