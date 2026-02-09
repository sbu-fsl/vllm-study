import csv
from typing import List, Dict

from src.dataset import Dataset


class CSVDataset(Dataset):
    def __init__(self, address: str, batch_size: int = 10):
        super().__init__(address)
        self.batch_size = batch_size

        self._total = -1

        self._file = open(self._addr, newline="", encoding="utf-8")
        self._reader = csv.DictReader(self._file)

        self._buffer: List[Dict] = []
        self._eof = False

    def _load_next_batch(self):
        """Load the next batch into memory."""
        self._buffer = []

        try:
            for _ in range(self.batch_size):
                row = next(self._reader)
                self._buffer.append(row)
        except StopIteration:
            self._eof = True

    def next(self) -> Dict:
        if not self._buffer and not self._eof:
            self._load_next_batch()

        if not self._buffer:
            self._file.close()
            raise StopIteration

        return self._buffer.pop(0)

    def count(self) -> int:
        """Counts rows without loading entire file into RAM."""
        if self._total == -1:
            with open(self._addr, newline="", encoding="utf-8") as f:
                self._total = sum(1 for _ in f) - 1  # minus header

        return self._total
