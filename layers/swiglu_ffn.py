from xformers.ops import SwiGLU

class SwiGLUFFN(SwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features = None,
        out_features = None,
        bias = True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )