from abc import ABC, abstractmethod


class FoundationalModel(ABC):
    """
    Base class for models served by the baseten.models package.
    """

    @property
    @abstractmethod
    def status(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass
