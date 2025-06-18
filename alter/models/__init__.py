from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .LRBGT import BrainNetworkTransformer


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
