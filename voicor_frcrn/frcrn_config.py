from transformers import PretrainedConfig, AutoConfig
from typing import List

class FRCRNConfig(PretrainedConfig):
    model_type = "frcrn"
    
    def __init__(
        self,
        model_dir: str = None,
        complex: bool = True,
        model_complexity: int = 45,
        model_depth: int = 14,
        log_amp: bool = False,
        padding_mode: str = "zeros",
        win_len: int = 640,
        win_inc: int = 320,
        fft_len: int = 640,
        win_type: str = "hann",
        **kwargs,
    ):
        self.model_dir = model_dir
        self.complex = complex
        self.model_complexity = model_complexity
        self.model_depth = model_depth
        self.log_amp = log_amp
        self.padding_mode = padding_mode
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        super().__init__(**kwargs)
        

AutoConfig.register("frcrn", FRCRNConfig)