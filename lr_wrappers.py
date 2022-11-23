import math
from paddle.optimizer.lr import LRScheduler
import paddle.optimizer.lr as lr


class WarmupLR(LRScheduler):
    def __init__(self, scheduler,
                 init_lr=1e-3,
                 num_warmup=1,
                 warmup_strategy='linear'):

        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0

        self._set_warmup_strategy(warmup_strategy)

        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def _set_warmup_strategy(self, warmup_strategy):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of " \
                             "['linear', 'cos', 'constant'] but got {}".format(warmup_strategy))

        # Define the strategy to warm up learning rate
        self._warmup_strategy = warmup_strategy

        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != '_scheduler')}
        wrapped_state_dict = {key: value for key, value in self._scheduler.__dict__.items()}

        # fix bug: https://github.com/DrRyanHuang/paddle-warmup-lr/issues/1
        wrapper_state_dict.pop("_warmup_func")

        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        # fix bug: https://github.com/DrRyanHuang/paddle-warmup-lr/issues/1
        old_choice = self._warmup_strategy  # 原来的选择

        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])

        if old_choice == self._warmup_strategy:
            pass
        else:
            self._set_warmup_strategy(self._warmup_strategy)

    def _format_param(self):

        try:
            lr = self._scheduler.get_lr()
        except NotImplementedError:
            # ReduceOnPlateau 没有 get_lr()
            lr = self._scheduler.__dict__['last_lr']

        self._scheduler.__dict__['warmup_max_lr'] = lr
        self._scheduler.__dict__['warmup_initial_lr'] = min(
            self._init_lr,
            lr
        )

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start

    def get_lr(self):
        step_num = self._step_count
        # warm up learning rate
        if step_num <= self._num_warmup:

            computed_lr = self._warmup_func(
                self._scheduler.__dict__['warmup_initial_lr'],
                self._scheduler.__dict__['warmup_max_lr'],
                step_num / self._num_warmup)

            lr = computed_lr
        else:
            lr = self._scheduler.get_lr()

        return lr

    def step(self, *args):
        self._scheduler.step(*args)

    def warmup_step(self):
        if self._step_count <= self._num_warmup:
            lr = self.get_lr()
            self._scheduler.__dict__['last_lr'] = lr
            self._step_count += 1

    def state_keys(self):
        self.keys = []
