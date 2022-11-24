from paddle.optimizer.lr import LRScheduler


class FixedLRScheduler(LRScheduler):
    def __init__(self, lr, warmup_steps=10):
        self._cur_epoch = -1
        self._current_iter = -1
        self._warmup_steps = warmup_steps if warmup_steps > 0 else 1
        self._start_lr = 0
        super().__init__(lr)
        self.warmup()

    def step(self, epoch=None):
        self._cur_epoch = self._cur_epoch + 1 if epoch is None else epoch
        self.last_lr = self.base_lr * (0.2 ** int(self._cur_epoch / 10))
        # print(f"{epoch} epoch {self._cur_epoch} lr {self.last_lr}")

    def warmup(self):
        if self._current_iter <= self._warmup_steps:
            self.last_lr = self._start_lr + (self.base_lr - self._start_lr) * self._current_iter / self._warmup_steps
            self._current_iter += 1

    def get_lr(self):
        return self.last_lr

    def state_keys(self):
        self.keys = ['_cur_epoch', '_current_iter']
