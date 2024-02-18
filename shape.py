import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy


class Shape:
    def __init__(self, r):
        self.r = r

    def external_field(self, external=[0, 0, 1]):
        self.ext = torch.tensor(external, dtype=torch.float64, device="cuda")

    def lattice_shape(self, size, num_point, shape="sphere"):
        if shape == "sphere":
            radius = size[0]
            points = [
                (i, j, k)
                for i in range(-num_point, num_point + 1)
                for j in range(-num_point, num_point + 1)
                for k in range(-num_point, num_point + 1)
                if (i**2 + j**2 + k**2) <= num_point**2
            ]
            self.distance = radius / num_point
            self.vol = 4 * 3.141592653589793238 / 3 * radius**3
        elif shape == "ellipse":
            a, b, c = size
            m = max(size)
            points = [
                (i, j, k)
                for i in range(-num_point, num_point + 1)
                for j in range(-num_point, num_point + 1)
                for k in range(-num_point, num_point + 1)
                if (i / a) ** 2 + (j / b) ** 2 + (k / c) ** 2 <= (num_point / m) ** 2
            ]
            self.distance=m/num_point
            self.vol = 4 / 3 * 3.14159 * a * b * c
        elif shape == "rectangular prism":
            a, b, c = map(lambda x: x / 2, size)
            r = num_point / max(size)
            points = [
                (i, j, k)
                for i in range(-num_point, num_point + 1)
                if -a * r <= i <= a * r
                for j in range(-num_point, num_point + 1)
                if -b * r <= j <= b * r
                for k in range(-num_point, num_point + 1)
                if -c * r <= k <= c * r
            ]
            self.distance=1/r
            self.vol = a * b * c * 8
        elif shape == "triangular prism":
            a, b = size[:-1]
            c = size[-1] / 2
            r = num_point / max((a, b, c))
            points = [
                (i, j, k)
                for i in range(-num_point, num_point + 1)
                for j in range(-num_point, num_point + 1)
                if i >= 0 and j >= 0 and i / a + j / b <= r
                for k in range(-num_point, num_point + 1)
                if -c * r <= k <= c * r
            ]
            self.distance=1/r
            self.vol = a * b * c
        
        points = torch.tensor(points, dtype=torch.float64)
        self.num_points = points.shape[0]
        self.dist_matrix = torch.cdist(points, points, p=2)
        self.dist_matrix[self.dist_matrix != 1] = 0
        self.lattice = torch.cat(
            (points * self.distance, torch.zeros((self.num_points, 7))), dim=1
        ).to("cuda")
        self.validnum_points = (self.dist_matrix.sum(dim=1) == 6).sum()

    def calculate_potential(self):
        for i in range(self.num_points):
            other = torch.cat([self.lattice[:i], self.lattice[i + 1 :]], dim=0)
            other[:, :3] = self.lattice[i, :3] - other[:, :3]
            norm = torch.norm(other[:, :3], dim=1) ** 3
            self.lattice[i, -1] = (
                ((other[:, 3:6] * other[:, :3]).sum(dim=1) / norm).sum()
                / 4
                / 3.141592653589793238
            )
        self.lattice[:, -1] -= torch.matmul(self.lattice[:, :3], self.ext)

    def update_polar(self, alpha):
        temp = copy.deepcopy(self.lattice[:, 3:6])
        for i in range(self.num_points):
            if torch.sum(self.dist_matrix[i]) == 6:
                near = self.lattice[self.dist_matrix[i] == 1].T
                E = -(
                    (near[-1, :] * ((near[:3, :].T - self.lattice[i, :3]).T)).sum(dim=1)
                    / 2
                    / self.distance**2
                )
                self.lattice[i, 3:6] = (
                    (self.r - 1) * E * self.vol / self.validnum_points
                )
                self.lattice[i, 6:9] = E
        self.lattice[:, 3:6] = (1 - alpha) * temp + alpha * self.lattice[:, 3:6]

    def iterate(self, epochs, alpha=1):
        compare = []
        for epoch in range(epochs):
            if epoch > epochs - 5:
                temp = self.lattice[:, 3:6]
                row = ~torch.all(temp == 0, dim=1)
                temp = temp[row]
            self.calculate_potential()
            self.update_polar(alpha)
            if epoch > epochs - 5:
                a = self.lattice[:, 3:6]
                a = a[row]
                compare.append(
                    (torch.norm(a - temp, dim=1) / torch.norm(temp, dim=1)).max().item()
                )
        return compare

    def plot_vectors(self, scale=1, showpercentage=0.1):
        """
        绘制向量图。
        """

        # 分离起点和方向
        self.lattice = self.lattice.to("cpu")
        start_points = self.lattice[:, :3]  # 前三列是起点
        directions = self.lattice[:, 3:6] * scale  # 后三列是方向

        # 创建3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 绘制每个向量
        for start, direction in random.sample(
            list(zip(start_points, directions)),
            int(round(self.num_points * showpercentage)),
        ):
            ax.quiver(
                start[0], start[1], start[2], direction[0], direction[1], direction[2]
            )

        # 设置坐标轴标签
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        # 显示图形
        plt.show()
        self.lattice = self.lattice.to("cuda")


def demo(params, epsilonr, epochs):
    global d
    d = Shape(epsilonr)
    if params == "sphere":
        d.lattice_shape([4], 12, "sphere")
    elif params == "rectangular prism":
        d.lattice_shape([6, 6, 6], 17, "rectangular prism")
    elif params == "ellipse1":
        d.lattice_shape([3, 3, 6], 18, "ellipse")
    elif params == "ellipse2":
        d.lattice_shape([3, 6, 3], 18, "ellipse")
    elif params == "triangular prism":
        d.lattice_shape([3, 3, 6], 22, "triangular prism")
    d.external_field()
    com = d.iterate(epochs)
    print(
        f"the maxium relative changes of polarization in last 4 epochs are {max(com)}"
    )
    p = torch.norm(d.lattice[:, 3:6].sum(dim=0)) / d.vol
    p = p.to("cpu")
    d.dist_matrix.sum
    ind = torch.sum(d.dist_matrix, dim=1) == 6
    ep = d.lattice[ind, 6:9].mean(dim=0).to("cpu") - torch.tensor([0, 0, 1])
    ep = torch.norm(ep)
    return (ep / p).item()
