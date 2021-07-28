from ...configuration_utils import PretrainedConfig


class EdsrConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~super_image.EdsrModel`.
        It is used to instantiate the model according to the specified arguments, defining the model architecture.
        Instantiating a configuration with the defaults will yield a similar
        configuration to that of the [EDSR base](https://huggingface.co/eugenesiow/edsr-base) architecture.
        Configuration objects inherit from :class:`~super_image.PretrainedConfig` and can be used to control the model
        outputs. Read the documentation from :class:`~super_image.PretrainedConfig` for more information.
        Examples:
            ```python
            from super_image import EdsrModel, EdsrConfig
            # Initializing a configuration
            config = EdsrConfig(
                scale=4,                                # train a model to upscale 4x
            )
            # Initializing a model from the configuration
            model = EdsrModel(config)
            # Accessing the model configuration
            configuration = model.config
            ```
        """
    model_type = 'EDSR'

    def __init__(self, scale: int = None, n_resblocks=16, n_feats=64, n_colors=3, rgb_range=255,
                 res_scale=1, data_parallel=False, **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_resblocks (int): Number of residual blocks.
            n_feats (int): Number of features.
            n_colors (int):
                Number of color channels.
            rgb_range (int):
                Range of RGB as a multiplier to the MeanShift.
            res_scale (int):
                The res scale multiplier.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.data_parallel = data_parallel
