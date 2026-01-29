from collections import deque
import random


class ReplayMemory(object):
    def __init__(self, max_len):
        super().__init__()
        self.memory = deque(maxlen=max_len)

    def push_transition(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_len(self):
        return len(self.memory)
