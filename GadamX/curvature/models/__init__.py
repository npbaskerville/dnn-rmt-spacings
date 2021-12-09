from .preresnet import *
from .vgg import *
from .vggleaky import *
from .vggtest import *
from .vgg2 import *
from .wide_resnet import *
from .resnet import *
from .vgg2 import *
from .vgg_drop import *

#Added by Diego 24/03/2020
from .MLP import *

# Added by Xingchen Wan 20 Oct
from .resnext2 import *
from .densenet import *

# Added by Xingchen Wan 2 Dec
from .all_cnn import *

# Added by Xingchen on 23 Dec - the RNN/LSTM model for NLP tasks
from .language_model import *

# Added by Xingchen on 26 Dec - logistic regression toy example
from .logistic_regression import *

# Added by Xingchen on 1 Jan - added Shake-shake architecture
from .shakeshake import *

from .lenet import LeNet

import torchvision.models as modelstorch
resnet50 = modelstorch.resnet50(pretrained=False)
resnet18 = modelstorch.resnet18()
