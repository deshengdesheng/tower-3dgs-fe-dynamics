# Modified from upstream gsplat (Apache-2.0) by the authors of this release:
# export the authors' region-filtered training strategies.

from .base import Strategy
from .default import DefaultStrategy
from .irregular_region_strategy import IrregularRegionFilteredStrategy
from .mcmc import MCMCStrategy
from .region_strategy import RegionFilteredStrategy
