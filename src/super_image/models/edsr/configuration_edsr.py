from ...configuration_utils import PretrainedConfig


class EdsrConfig(PretrainedConfig):
    model_type = 'EDSR'

    def __init__(self, scale=None, n_resblocks=16, n_feats=64, n_colors=3, rgb_range=255,
                 res_scale=1, data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.data_parallel = data_parallel
