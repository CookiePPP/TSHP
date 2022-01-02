import torch
from TSHP.utils.modules.core import nnModule
from TSHP.models.classifiers.ncwn_spkr_SIMC.model import Model as modelmodule

class SpeakerEncoder(nnModule):
    def __init__(self, path):
        super().__init__()
        self.model, *_ = modelmodule.load_model(path, train=False)
        self.model_identifier = sum(x.std()-x.mean() for x in self.model.parameters()).item()
    
    def forward(self, mel, lens):
        assert mel.dim() == 3
        assert lens is None or lens.dim() == 3
        with torch.no_grad():
            spkr_embed = self.model.get_embed(mel, lens).data.clone()
        return spkr_embed # [B, 1, C]