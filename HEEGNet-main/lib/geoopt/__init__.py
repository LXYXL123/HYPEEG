from . import manifolds
from . import tensor
from . import linalg
from . import utils
from .utils import ismanifold

# The vendored geoopt package eagerly imports optional subpackages that are
# not needed for our Lorentz-manifold bridge. Newer SciPy versions removed
# some legacy symbols used by geoopt.optim.rlinesearch, which would otherwise
# make even `from lib.geoopt import Lorentz` fail. Keep manifold/tensor access
# available and degrade optional modules gracefully.
try:
    from . import optim
except Exception:  # pragma: no cover - optional dependency path
    optim = None

try:
    from . import samplers
except Exception:  # pragma: no cover - optional dependency path
    samplers = None

from .tensor import ManifoldParameter, ManifoldTensor
from .manifolds import (
    Manifold,
    Stiefel,
    EuclideanStiefelExact,
    CanonicalStiefel,
    EuclideanStiefel,
    Euclidean,
    Sphere,
    SphereExact,
    PoincareBall,
    PoincareBallExact,
    Stereographic,
    StereographicExact,
    SphereProjection,
    SphereProjectionExact,
    ProductManifold,
    StereographicProductManifold,
    Scaled,
    Lorentz,
    BirkhoffPolytope,
    SymmetricPositiveDefinite,
    UpperHalf,
    BoundedDomain,
)

__version__ = "0.5.0"
