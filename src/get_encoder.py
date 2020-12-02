import re,copy
import torch.utils.model_zoo as model_zoo
def get_encoder(name, in_channels=3, depth=5, weights=None,encoders_ =None):
    encoders = copy.deepcopy(encoders_)
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder
