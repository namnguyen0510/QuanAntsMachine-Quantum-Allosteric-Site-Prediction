"""Allosteric-prediction methods.

All methods implement the :class:`AllosteryMethod` interface and can be looked
up by string name through :data:`METHOD_REGISTRY`.

>>> from quanallo.methods import METHOD_REGISTRY
>>> method = METHOD_REGISTRY['dqaw_lifetime']()    # instantiate by name
>>> scores = method.compute(graph)
"""
from quanallo.methods.base import AllosteryMethod
from quanallo.methods.qsvd import QSVD
from quanallo.methods.dqaw import DQAWTimeAvg, DQAWLifetime
from quanallo.methods.qpagerank import QPageRank
from quanallo.methods.heatkernel import HeatKernel, CTQW
from quanallo.methods.commute import CommuteTime, GNM
from quanallo.methods.meta_learner import MetaLearner

#: Map from short method-name (str) to method class. Used by the high-level
#: :class:`AllostericPredictor` API for dispatch.
METHOD_REGISTRY: dict[str, type[AllosteryMethod]] = {
    "qsvd":           QSVD,
    "dqaw_timeavg":   DQAWTimeAvg,
    "dqaw_lifetime":  DQAWLifetime,
    "qpagerank":      QPageRank,
    "heatkernel":     HeatKernel,
    "ctqw":           CTQW,
    "commute_time":   CommuteTime,
    "gnm":            GNM,
    "meta_learner":   MetaLearner,
}

__all__ = [
    "AllosteryMethod",
    "QSVD",
    "DQAWTimeAvg",
    "DQAWLifetime",
    "QPageRank",
    "HeatKernel",
    "CTQW",
    "CommuteTime",
    "GNM",
    "MetaLearner",
    "METHOD_REGISTRY",
]
