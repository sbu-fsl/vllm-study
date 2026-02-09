from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, model: str):
        self._model = model

    def model(self) -> str:
        return self._model
    
    @abstractmethod
    def payload(self, prompt: str, opts: dict) -> dict:
        pass
