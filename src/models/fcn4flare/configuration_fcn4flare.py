from transformers.configuration_utils import PretrainedConfig


class FCN4FlareConfig(PretrainedConfig):
    """
    Configuration class for FCN4Flare model.
    """
    model_type = "fcn4flare"

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        output_dim=1,
        depth=4,
        dilation=[1, 2, 4, 8],
        maskdice_threshold=0.5,
        dropout_rate=0.1,
        kernel_size=3,
        **kwargs
    ):
        """Initialize FCN4FlareConfig."""
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.dilation = dilation
        self.maskdice_threshold = maskdice_threshold
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
