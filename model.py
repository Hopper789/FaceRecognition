from torch import nn
from torch.nn.functional import normalize, softmax
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict

class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.resnet = None
        self.hidden_layer = None
        self.arcFacePart = None
        self.loss_fn = None
    def init(self, loss, embedding_size = 512, hidden_size = 512):
        self.resnet = InceptionResnetV1(pretrained='vggface2', num_classes=512, classify=False)
        self.resnet.logits = nn.Linear(in_features=512, out_features=hidden_size)
        self.hidden_layer = nn.Sequential(
            OrderedDict(
                [
                    ('linear1', nn.Linear(hidden_size, hidden_size)),
                    ('non_lin1', nn.Tanh()),
                    ('dropout1', nn.Dropout(p=0.5)),

                    ('linear2', nn.Linear(hidden_size, hidden_size)),
                    ('non_lin2', nn.Tanh()),
                    ('dropout2', nn.Dropout(p=0.5))
                ]
            )
        )
        self.arcFacePart = nn.Linear(hidden_size, embedding_size, bias=False)
        self.loss_fn = loss
        nn.utils.parametrizations.weight_norm(self.arcFacePart)

    def encode(self, x):
        tt = self.hidden_layer(self.resnet(x))
        return normalize(self.arcFacePart(tt), p=2, dim=1)

    def get_logits(self, x):
        x = self.encode(x)
        return self.loss_fn.get_logits(x)

    def predict_proba(self, x):
        return softmax(self.get_logits(x))

