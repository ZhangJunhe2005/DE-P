import torch
import torch.nn as nn


class DepHead(nn.Module):
    VARIANTS = ("unified", "split")

    def __init__(self, input_dim, output_dim, variant="unified"):
        super(DepHead, self).__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"head variant must be one of {self.VARIANTS}")
        if output_dim != 10:
            raise ValueError("DEP head requires 9 endstate channels and one score channel")
        self.variant = variant
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        if variant == "unified":
            self.model.append(nn.Conv2d(256, output_dim, kernel_size=1, stride=1))
        else:
            self.trajectory_head = nn.Conv2d(256, 9, kernel_size=1, stride=1)
            self.score_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)

    def forward(self, x):
        output = self.model(x)
        if self.variant == "unified":
            return output
        return torch.cat(
            (self.trajectory_head(output), self.score_head(output)), dim=1
        )

    def score_parameters(self):
        if self.variant != "split":
            raise RuntimeError("independent score parameters require the split head")
        return self.score_head.parameters()

    def candidate_parameters(self):
        if self.variant != "split":
            raise RuntimeError("independent candidate parameters require the split head")
        yield from self.model.parameters()
        yield from self.trajectory_head.parameters()
