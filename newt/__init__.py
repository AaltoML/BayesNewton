from .kernels import *
from .utils import *
from .ops import *
from .likelihoods import *
from .models import *
from .basemodels import *
from .inference import *
from .cubature import *


def build_model(model, inf, name='GPModel'):
    return type(name, (inf, model), {})
