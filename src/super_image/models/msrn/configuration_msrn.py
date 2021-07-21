from ...configuration_utils import PretrainedConfig


class MsrnConfig(PretrainedConfig):
    model_type = 'MSRN'

    def __init__(self, scale=None, n_blocks=8, n_feats=64, rgb_range=255, bam=False,
                 data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.rgb_range = rgb_range
        self.data_parallel = data_parallel
        self.bam = bam
