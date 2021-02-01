from torch.optim.lr_scheduler import ReduceLROnPlateau


class Scheduler(ReduceLROnPlateau):
    def step(self, metrics, epoch=None, pbar=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        current_is_better = False
        new_lrs = []

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            current_is_better = True
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            new_lrs = self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return current_is_better, new_lrs

    def _reduce_lr(self, epoch):
        new_lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                new_lrs.append(new_lr)
        return new_lrs
