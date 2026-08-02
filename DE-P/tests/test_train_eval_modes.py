import types
import unittest

import torch

from policy.dep_trainer import DepTrainer


class _Console:
    def log(self, *args, **kwargs):
        pass


class _Progress:
    def __init__(self):
        self.console = _Console()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def add_task(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        pass

    def remove_task(self, *args, **kwargs):
        pass


class _Tensorboard:
    def add_scalar(self, *args, **kwargs):
        pass


class TrainEvalModeTests(unittest.TestCase):
    def _trainer_for_eval(self):
        trainer = DepTrainer.__new__(DepTrainer)
        trainer.policy = torch.nn.Sequential(torch.nn.BatchNorm2d(1), torch.nn.Conv2d(1, 1, 1))
        trainer.policy.train()
        trainer.batch_size = 2
        trainer.loss_weight = [1.0, 1.0]
        trainer.progress_log = _Progress()
        trainer.tensorboard_log = _Tensorboard()
        depth = torch.randn(2, 1, 4, 4)
        dummy = torch.zeros(2, 1)
        trainer.val_dataloader = [(depth, dummy, dummy, dummy, dummy)]

        def fake_forward(self, depth, *_):
            value = self.policy(depth).mean()
            zero = value * 0.0
            return value, value, zero, zero, zero

        trainer.forward_and_compute_loss = types.MethodType(fake_forward, trainer)
        return trainer

    def test_validation_sets_eval_and_preserves_batchnorm_statistics(self):
        trainer = self._trainer_for_eval()
        bn = next(module for module in trainer.policy.modules() if isinstance(module, torch.nn.BatchNorm2d))
        before = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
        trainer.eval_one_epoch(0)
        after = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
        self.assertTrue(all(not module.training for module in trainer.policy.modules()
                            if isinstance(module, torch.nn.BatchNorm2d)))
        for lhs, rhs in zip(before, after):
            self.assertTrue(torch.equal(lhs, rhs))

    def test_each_epoch_restores_train_before_training_and_uses_eval_for_validation(self):
        trainer = DepTrainer.__new__(DepTrainer)
        trainer.policy = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
        trainer.progress_log = _Progress()
        train_modes = []
        eval_modes = []

        def fake_train_one_epoch(epoch, task):
            train_modes.append(trainer.policy.training)

        def fake_eval_one_epoch(epoch):
            trainer.policy.eval()
            eval_modes.append(trainer.policy.training)

        trainer.train_one_epoch = fake_train_one_epoch
        trainer.eval_one_epoch = fake_eval_one_epoch
        trainer.train(epoch=2)
        self.assertEqual(train_modes, [True, True])
        self.assertEqual(eval_modes, [False, False])

    def test_late_freeze_keeps_only_frozen_batchnorm_in_eval(self):
        trainer = DepTrainer.__new__(DepTrainer)
        frozen = torch.nn.BatchNorm2d(1)
        trainable = torch.nn.BatchNorm2d(1)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        backbone = torch.nn.Module()
        backbone.image_backbone = torch.nn.Sequential(frozen, trainable)
        trainer.policy = backbone
        trainer.freeze_policy = "late"
        trainer.set_training_mode()
        self.assertFalse(frozen.training)
        self.assertTrue(trainable.training)


if __name__ == "__main__":
    unittest.main()
