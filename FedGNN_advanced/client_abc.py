from abc import ABC, abstractmethod

class FederatedClient(ABC):
    @abstractmethod
    def train_local(self, global_state, c_global=None):
        pass

    @abstractmethod
    def evaluate(self, model_state_dict=None, mask=None):
        pass
