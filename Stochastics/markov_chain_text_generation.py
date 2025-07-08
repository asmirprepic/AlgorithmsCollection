import random
import re
from collections import defaultdict

class MarkovChainTextGenerator:
    def __init__(self,n=1):
        self.n = n,
        self.model = defaultdict(list)

    def preprocess(self,text):
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().split()

    def train(self, text):
        words = self.preprocess(text)
        if len(words) < self.n + 1:
            raise ValueError("To short.")

        for i in range(len(words) - self.n):
            prefix = tuple(words[i:i + self.n])
            next_word = words[i + self.n]
            self.model[prefix].append(next_word)

    def generate(self, length=50):
        if not self.model:
            raise ValueError("Train before generation.")

        start = random.choice(list(self.model.keys()))
        output = list(start)

        for _ in range(length - self.n):
            prefix = tuple(output[-self.n:])
            next_words = self.model.get(prefix)
            if not next_words:
                break
            output.append(random.choice(next_words))

        return ' '.join(output)
