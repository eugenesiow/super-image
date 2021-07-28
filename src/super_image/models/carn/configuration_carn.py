from ...configuration_utils import PretrainedConfig


class CarnConfig(PretrainedConfig):
    model_type = 'CARN'

    def __init__(self, scale=None, bam=False, data_parallel=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.bam = bam
        self.data_parallel = data_parallel
