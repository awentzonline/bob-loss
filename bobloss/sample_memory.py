import numpy as np


class SampleMemory(object):
    def __init__(self, item_shape, max_size):
        self.memory = np.zeros((max_size,) + item_shape)
        self.item_shape = item_shape
        self.num_stored = 0
        self.max_size = max_size
        self.tail_index = 0

    def sample(self, num_samples):
        indexes = self.sample_indexes(num_samples)
        return self.memory[indexes]

    def sample_indexes(self, num_samples):
        return np.random.randint(
            0, self.num_stored, (num_samples,)
        )

    def append(self, item):
        self.memory[self.tail_index, :] = item
        self.tail_index = (self.tail_index + 1) % self.max_size
        self.num_stored = min(self.num_stored + 1, self.max_size)

    def append_batch(self, batch):
        batch_size = batch.shape[0]
        batch_tail_index = self.tail_index + batch_size
        wrap_extra = batch_tail_index - self.max_size
        chunk_tail_index = min(batch_tail_index, self.max_size)
        self.memory[self.tail_index:chunk_tail_index, :] = batch[:batch_size - wrap_extra]
        if wrap_extra > 0:
            self.memory[:wrap_extra, :] = batch[batch_size - wrap_extra:]
        self.tail_index = batch_tail_index % self.max_size
        self.num_stored = min(self.num_stored + batch_size, self.max_size)

    def last_n_frames(self, n):
        return self.memory[self.last_n_frames_indexes(n)]

    def last_n_frames_indexes(self, n):
        tail = self.tail_index
        start = tail - n
        indexes = range(max(start, 0), tail)
        if start < 0:
            indexes = range(self.num_stored + start, self.num_stored) + indexes
        return indexes


if __name__ == '__main__':
    shape = (32, 32, 3)
    max_size = 100
    for num_items, sample_size in ((10, 5), (10, 10), (100, 32), (120, 10)):
        mem = SampleMemory(shape, max_size)
        assert(mem.num_stored == 0)
        assert(mem.max_size == max_size)
        assert(mem.memory.shape == (max_size,) + shape)
        for i in range(num_items):
            mem.append(np.random.random(shape))
        assert(mem.tail_index == num_items % max_size)
        assert(mem.num_stored == min(num_items, max_size))
        indexes = mem.sample_indexes(sample_size)
        assert(indexes.shape[0] == sample_size)
        assert(indexes.min() >= 0)
        assert(indexes.max() < num_items)
        samples = mem.sample(sample_size)
        assert(samples.shape == (sample_size,) + shape)
    mem = SampleMemory(shape, max_size)
    batch_size = 10
    batch = np.random.random((batch_size,) + shape)
    mem.append_batch(batch)
    assert(mem.num_stored == batch_size)
    assert(mem.tail_index == batch_size)
    assert(np.array_equal(mem.memory[:5], batch[:5]))
    batch_size = 100
    batch = np.random.random((batch_size,) + shape)
    mem.append_batch(batch)
    assert(mem.num_stored == max_size)
    assert(mem.tail_index == 10)
    assert(np.array_equal(mem.memory[:5], batch[-10:-5]))
    assert(mem.last_n_frames_indexes(15) == map(lambda x: x % 100, range(95, 110)))
