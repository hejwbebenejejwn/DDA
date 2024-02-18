import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy

pi = np.pi


class sphere:
    def __init__(self, rel, radius, num_theta, num_radius, num_phi, device="cpu"):
        """unit:(r,the,pz,Ez)"""
        self.rel = rel
        self.radius = radius
        thetas = torch.linspace(0, pi / 2, num_theta + 1)[:-1] + pi / 2 / num_theta / 2
        radiuses = torch.linspace(0, radius, num_radius + 1)[1:]
        rad, the = torch.meshgrid(radiuses, thetas)
        flatrad, flatthe = rad.flatten(), the.flatten()
        self.unit = torch.stack((flatrad, flatthe), dim=1)
        self.unit = torch.cat((self.unit, torch.zeros_like(self.unit)), dim=1).to(
            device=device
        )
        self.phis = torch.linspace(0, pi, num_phi + 1)[:-1] + pi / 2 / num_phi / 2
        self.numpoint = 4 * num_radius * num_theta * num_phi
        self.num_phi = num_phi
        self.unitvol = (
            pi
            * radius**3
            / 3
            / num_phi
            / torch.sum(self.unit[:, 0] ** 2 * torch.sin(self.unit[:, 1]))
        )
        self.rpos3 = []
        self.rpos5 = []
        self.rneg3 = []
        self.rneg5 = []
        dz2 = (
            (self.unit[:, 0] * torch.cos(self.unit[:, 1])).unsqueeze(1)
            - self.unit[:, 0] * torch.cos(self.unit[:, 1])
        ) ** 2
        benchmark = torch.stack(
            (
                self.unit[:, 0] * torch.cos(self.unit[:, 1]),
                self.unit[:, 0] * torch.sin(self.unit[:, 1]),
                torch.zeros_like(self.unit[:, 0]),
            ),
            dim=1,
        ).double()
        for phi in self.phis:
            pos = torch.stack(
                (
                    self.unit[:, 0] * torch.cos(self.unit[:, 1]),
                    self.unit[:, 0] * torch.sin(self.unit[:, 1]) * torch.cos(phi),
                    self.unit[:, 0] * torch.sin(self.unit[:, 1]) * torch.sin(phi),
                ),
                dim=1,
            ).double()
            neg = torch.stack(
                (
                    -self.unit[:, 0] * torch.cos(self.unit[:, 1]),
                    self.unit[:, 0] * torch.sin(self.unit[:, 1]) * torch.cos(phi),
                    self.unit[:, 0] * torch.sin(self.unit[:, 1]) * torch.sin(phi),
                ),
                dim=1,
            ).double()
            rpos = torch.cdist(benchmark, pos)
            rneg = torch.cdist(benchmark, neg)
            self.rpos3.append((rpos**3).to(device=device))
            self.rneg3.append((rneg**3).to(device=device))
            self.rpos5.append((dz2 / rpos**5).to(device=device))
            self.rneg5.append((dz2 / rneg**5).to(device=device))
        self.unit[:, -1] = 1

    def update_polar(self, alpha=1):
        r2 = self.unit[:, 0] ** 2
        sintheta = torch.sin(self.unit[:, 1])
        self.unit[:, 2] = (
            self.unitvol * (self.rel - 1) * self.unit[:, 3] * r2 * sintheta
        ) * alpha + self.unit[:, 2] * (1 - alpha)

    def cal_field(self):
        self.unit[:, -1] = 0
        p = self.unit[:, 2]
        for ind, phi in enumerate(self.phis):
            r3 = self.rpos3[ind]
            r5 = self.rpos5[ind]
            self.unit[:, -1] -= torch.sum(p / r3, dim=1)
            self.unit[:, -1] += 3 * torch.sum(p * r5, dim=1)
            r3 = self.rneg3[ind]
            r5 = self.rneg5[ind]
            self.unit[:, -1] -= torch.sum(p / r3, dim=1)
            self.unit[:, -1] += 3 * torch.sum(p * r5, dim=1)
        self.unit[:, -1] /= 2 * np.pi
        self.unit[:, -1] += 1

    def iterate(self, epochs, alpha=1, decend=False):
        if not decend:
            changes = []
            for epoch in range(epochs):
                temp = self.unit[:, -2].sum().detach().item()
                self.update_polar(alpha=alpha)
                self.cal_field()
                if epoch > epochs - 5:
                    if temp == 0:
                        changes.append(np.nan)
                    else:
                        changes.append(
                            abs(self.unit[:, -2].sum().detach().item() - temp)
                            / abs(temp)
                        )
                    totalpol = self.unit[:, -2].sum().detach().item()
            return totalpol * 4 * self.num_phi, changes
        else:
            lastchange = 10
            newchange = 9
            self.update_polar(alpha=alpha)
            self.cal_field()
            while newchange < lastchange:
                lastchange = newchange
                temp = self.unit[:, -2].sum().detach().item()
                self.update_polar(alpha=alpha)
                self.cal_field()
                newchange = abs(self.unit[:, -2].sum().detach().item() - temp) / abs(
                    temp
                )
            return temp * 4 * self.num_phi, lastchange


if __name__ == "__main__":
    sp = sphere(
        rel=1.1, radius=10, num_theta=15, num_radius=15, num_phi=15, device="cuda"
    )
    sp.iterate(10)
