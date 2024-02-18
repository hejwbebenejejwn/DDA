import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy


class DDA:
    def __init__(self, r):
        self.r = r

    def external_field(self, external):
        self.ext = torch.tensor(external, dtype=torch.float64, device="cuda")

    def lattice_sphere(self, radius, num_point):
        points = [
            (i, j, k)
            for i in range(-num_point, num_point + 1)
            for j in range(-num_point, num_point + 1)
            for k in range(-num_point, num_point + 1)
            if (i**2 + j**2 + k**2) <= num_point**2
        ]
        points = torch.tensor(points, dtype=torch.float64)
        self.num_points = points.shape[0]
        self.dist_matrix = torch.cdist(points, points, p=2)
        self.dist_matrix[self.dist_matrix != 1] = 0
        self.lattice = torch.cat(
            (points * radius / num_point, torch.zeros((self.num_points, 4))), dim=1
        ).to("cuda")
        self.distance = radius / num_point
        self.radius = radius
        self.validnum_points = (self.dist_matrix.sum(dim=1) == 6).sum()

    def calculate_potential(self):
        for i in range(self.num_points):
            other = torch.cat([self.lattice[:i], self.lattice[i + 1 :]], dim=0)
            other[:, :3] = self.lattice[i, :3] - other[:, :3]
            norm = torch.norm(other[:, :3], dim=1) ** 3
            self.lattice[i, -1] = (
                ((other[:, 3:6] * other[:, :3]).sum(dim=1) / norm).sum() / 4 / 3.1415926
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
                    (self.r - 1)
                    * E
                    * 4
                    * 3.1415926
                    * (self.radius) ** 3
                    / 3
                    / self.validnum_points
                )
        self.lattice[:, 3:6] = (1 - alpha) * temp + alpha * self.lattice[:, 3:6]

    def iterate(self, epochs, alpha):
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


if __name__ == "__main__":
    d = DDATorch(1.1)
    d.lattice_sphere(15, 10)
    d.external_field([0, 0, 1])

    start_time = time.time()

    com = d.iterate(50)

    end_time = time.time()
    print(f"运行时间: {end_time - start_time} 秒")

    print(
        f"the maxium relative changes of polarization in last 4 epochs are {max(com)}"
    )
    d.plot_vectors(scale=5, showpercentage=0.05)
    a = torch.norm(d.lattice[:, 3:6].sum(dim=0))
    b = 4 * 3.1415926 * d.radius**3 * (d.r - 1) / (d.r + 2) * torch.norm(d.ext)
    print(f"pest/e={a},pcal/e={b},relative error is {(a-b)/b}")
