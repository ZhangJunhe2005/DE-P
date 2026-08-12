import torch
import torch.nn as nn


class DepHead(nn.Module):
    VARIANTS = ("unified", "split", "independent")

    @staticmethod
    def _feature_tower(input_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def __init__(self, input_dim, output_dim, variant="unified"):
        super(DepHead, self).__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"head variant must be one of {self.VARIANTS}")
        if output_dim != 10:
            raise ValueError("DEP head requires 9 endstate channels and one score channel")
        self.variant = variant
        if variant == "independent":
            self.trajectory_model = self._feature_tower(input_dim)
            self.score_model = self._feature_tower(input_dim)
            self.trajectory_head = nn.Conv2d(256, 9, kernel_size=1, stride=1)
            self.score_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)
            return
        self.model = self._feature_tower(input_dim)
        if variant == "unified":
            self.model.append(nn.Conv2d(256, output_dim, kernel_size=1, stride=1))
        else:
            self.trajectory_head = nn.Conv2d(256, 9, kernel_size=1, stride=1)
            self.score_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)

    def forward(self, x):
        if self.variant == "unified":
            return self.model(x)
        if self.variant == "independent":
            return torch.cat((
                self.trajectory_head(self.trajectory_model(x)),
                self.score_head(self.score_model(x)),
            ), dim=1)
        output = self.model(x)
        return torch.cat(
            (self.trajectory_head(output), self.score_head(output)), dim=1
        )

    def score_parameters(self):
        if self.variant == "split":
            return self.score_head.parameters()
        if self.variant == "independent":
            return iter((*self.score_model.parameters(), *self.score_head.parameters()))
        raise RuntimeError("independent score parameters require a branched head")

    def candidate_parameters(self):
        if self.variant == "split":
            yield from self.model.parameters()
            yield from self.trajectory_head.parameters()
            return
        if self.variant == "independent":
            yield from self.trajectory_model.parameters()
            yield from self.trajectory_head.parameters()
            return
        raise RuntimeError("independent candidate parameters require a branched head")
