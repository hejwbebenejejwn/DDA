import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy


class DDAcomplex:
    def __init__(self, r):
        self.r = r

    def external_field(self, lambdaa, E0=1):
        self.k = 2 * 3.14159 / lambdaa
        self.dis_matexp = torch.exp(1j * self.k * self.dis_mat)
        self.dis_matexpd3 = self.dis_matexp / self.dis_mat3
        self.dis_matexpd5 = self.dis_matexp / self.dis_mat5
        self.dis_matexpd3 = self.dis_matexpd3.to("cuda")
        self.dis_matexpd5 = self.dis_matexpd5.to("cuda")
        self.dis_matexp = None
        self.dis_mat = None
        self.dis_mat3 = None
        self.dis_mat5 = None
        self.ext = E0 * torch.exp(1j * self.k * self.lattice[:, 0])

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
        self.dis_mat = torch.cdist(points, points, p=2)
        self.lattice = (
            torch.cat(
                (points * radius / num_point, torch.zeros((self.num_points, 6))), dim=1
            )
            .to(dtype=torch.complex128)
            .to("cuda")
        )
        self.distance = radius / num_point
        self.dis_mat *= self.distance
        self.dis_mat3 = self.dis_mat**3
        self.dis_mat5 = self.dis_mat**5
        self.radius = radius

    def calculate_field(self):
        for i in range(self.num_points):
            temp = torch.cat((self.lattice[:i, :6], self.lattice[i + 1 :, :6]), dim=0)
            temp[:, :3] *= -1
            temp[:, :3] += self.lattice[i, :3]
            disexpd3 = torch.cat(
                (self.dis_matexpd3[i, :i], self.dis_matexpd3[i, i + 1 :]), dim=0
            )
            disexpd5 = torch.cat(
                (self.dis_matexpd5[i, :i], self.dis_matexpd5[i, i + 1 :]), dim=0
            )

            self.lattice[i, 6:9] = (
                (
                    (
                        (
                            torch.bmm(
                                temp[:, :3].view(-1, 1, 3),
                                temp[:, 3:6].view(-1, 3, 1),
                            )
                        ).view(self.num_points - 1)
                    )
                    * disexpd5
                ).view(-1, 1)
                * temp[:, :3]
            ).sum(dim=0)
            self.lattice[i, 6:9] *= 3
            self.lattice[i, 6:9] -= ((disexpd3).view(-1, 1) * temp[:, 3:]).sum(dim=0)
        self.lattice[:, 6:9] /= 4 * 3.14159
        self.lattice[:, -1] += self.ext

    def update_polar(self, alpha):
        temp = copy.deepcopy(self.lattice[:, 3:6])
        self.lattice[:, 3:6] = (self.r - 1) * self.lattice[:, 6:]
        self.lattice[:, 3:6] *= (
            4 * 3.1415 / 3 * self.radius**3 / self.num_points / 1.2
        )
        self.lattice[:, 3:6] = self.lattice[:, 3:6] * alpha + temp * (1 - alpha)

    def iterate(self, epochs, alpha=1):
        compare = []
        for epoch in range(epochs):
            if epoch > epochs - 5:
                temp = copy.deepcopy(self.lattice[:, 3:6])
            self.calculate_field()
            self.update_polar(alpha)
            if epoch > epochs - 5:
                a = self.lattice[:, 3:6]
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
        start_points = torch.real(self.lattice[:, :3])  # 前三列是起点
        directions = torch.real(self.lattice[:, 3:6]) * scale  # 后三列是方向

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
    d = DDAcomplex(1.3)
    d.lattice_sphere(15, 10)
    d.external_field(lambdaa=100)
    com = d.iterate(20)
    print(
        f"the maxium relative changes of polarization in last 4 epochs are {max(com)}"
    )
    # d.plot_vectors(scale=10, showpercentage=0.05)
    a = torch.norm(d.lattice[:, 3:6].sum(dim=0))
    print(f"pest/e={a}")
