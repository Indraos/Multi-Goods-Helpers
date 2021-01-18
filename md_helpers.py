#!/usr/bin/env python3
"""Computes optimal mechanisms and dual solutions for the multi-goods monopolist problem.

The classes in this script allow to compute and visualize \
    optimal solutions to the multi-goods monopolist problem.

Example usage:

types = np.array([[0,0],[1,1]])
probabilities = np.array([1/2,1/2])
monopolist = MGMProblem(types, probabilities)
monopolist.solve()
monopolist.save("monopolist_viz.html")
"""
from itertools import permutations
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pulp as p
import mpld3
from mpld3 import plugins

matplotlib.use("Agg")


class MGMProblem:
    """Class for the Multi-Goods Monopolist Problem"""

    def __init__(self, types: np.array, probabilities: np.array = None):
        """Constructor

        Args:
            types (np.array): Array containing types
            probabilities (np.array, optional): contains probabilities. Defaults to None.
        """
        self._types = np.array(types)
        self._probabilities = np.array(probabilities)
        assert len(self._types) == len(
            self._probabilities
        ), "Size of support and dimension of probability distribution must match"
        assert np.isclose(
            self._probabilities.sum(), 1
        ), "Probability must be normalized"
        assert (self._probabilities >= 0).all(), "Probabilities must be positive."
        self._transfers_var = p.LpVariable.dicts(
            "Transfers", range(self.supp), cat="Continuous"
        )
        self.allocations_var = p.LpVariable.dicts(
            "Allocations",
            [(i, j) for i in range(self.supp) for j in range(self.dim)],
            lowBound=0,
            upBound=1,
            cat="Continuous",
        )
        self._problem = p.LpProblem("Bundling Problem", p.LpMaximize)
        # objective
        self._problem += p.lpSum(
            [
                self._probabilities[typ] * self._transfers_var[typ]
                for typ in range(self.supp)
            ]
        )
        # constraints
        for typ in range(self.supp):
            self._problem += p.LpConstraint(
                p.lpSum(
                    [
                        self.allocations_var[(typ, j)] * self._types[typ, j]
                        for j in range(self.dim)
                    ]
                )
                - self._transfers_var[typ],
                sense=p.LpConstraintGE,
                rhs=0,
                name="IR_{}".format(typ),
            )
            for dev in range(self.supp):
                if typ != dev:
                    self._problem += p.LpConstraint(
                        p.lpSum(
                            [
                                self.allocations_var[(typ, j)] * self._types[typ, j]
                                - self.allocations_var[(dev, j)] * self._types[typ, j]
                                for j in range(self.dim)
                            ]
                        )
                        - self._transfers_var[typ]
                        + self._transfers_var[dev],
                        rhs=0,
                        sense=p.LpConstraintGE,
                        name="IC_{}{}".format(typ, dev),
                    )

    @property
    def dim(self):
        """dimension of the problem

        Returns:
            int: dimension
        """
        return len(self._types[0])

    @property
    def supp(self):
        """support size of the type space

        Returns:
            int: support size
        """
        return len(self._types)

    @property
    def allocations(self):
        """allocations

        Returns:
            np.array: allocations for types
        """
        allocations = np.zeros((self.supp, self.dim))
        for i in range(self.supp):
            for j in range(self.dim):
                allocations[i, j] = self.allocations_var[(i, j)].value()
        return allocations

    @property
    def transfers(self):
        """transfers

        Returns:
            np.array: transfers for types
        """
        return [transfer.value() for transfer in self._transfers_var.values()]

    @property
    def revenue(self):
        """revenue of the mechanism

        Returns:
            np.float: revenue of the mchanism
        """
        return p.value(self._problem.objective)

    @property
    def constraints(self):
        """Return constraints of the problem

        Returns:
            [p.LpConstraint]: List of Constraints of the problem.
        """
        return self._problem.constraints.items()

    @property
    def lam(self):
        """Dual variables of the optimal solution

        Returns:
            np.array: dual variables of the problem
        """

        ic_duals = {
            name: c.pi for name, c in self.constraints if (not name.startswith("IR"))
        }
        lam = np.zeros((self.supp, self.supp))
        for i in range(self.supp):
            for j in range(self.supp):
                if i != j:
                    lam[i, j] = ic_duals["IC_{}{}".format(i, j)]
        return lam

    @property
    def initial_lam(self):
        """Candidate initial dual variables as in Myerson '81

        Returns:
            np.array: initial dual variables
        """
        initial_lam = np.zeros((self.supp, self.supp))
        for i in range(self.supp):
            for j in range(self.supp):
                initial_lam[i, j] = (
                    (1 - np.cumsum(self._probabilities)[i - 1]) if j + 1 == i else 0
                )
        return initial_lam

    @property
    def virtual(self):
        """virtual values associated to optimal dual variables

        Returns:
            np.array: virtual values for agents and types
        """
        return self._virtual_from_flow(self.lam)

    @property
    def initial_virtual(self):
        """virtual values associated with the initial dual variables

        Returns:
            np.array: initial virtual values for agents and types
        """
        return self._virtual_from_flow(self.initial_lam)

    def _virtual_from_flow(self, lam):
        """utility that computes virtual values for given dual variables

        Args:
            lam (np.array): dual variables

        Returns:
            np.array: virtual values
        """
        virtual_value = self._types.copy()
        for i in range(self.supp):
            for j in range(self.supp):
                virtual_value[i] -= (
                    lam[i, j]
                    * (self._types[i] - self._types[j])
                    / self._probabilities[i]
                )
        return virtual_value

    def solve(self):
        """solve Linear Program"""
        self._problem.solve()

    def is_grand_bundling(self):
        """Determines whether solution is grand bundling

        Returns:
            bool: grand bundling
        """
        for allocation in self.allocations:
            if not (
                np.isclose(allocation, np.zeros_like(allocation)).all()
                or np.isclose(allocation, np.ones_like(allocation)).all()
            ):
                return False
        return True

    def is_upgrade_pricing(self):
        """Determines whether solution is upgrade pricing

        Returns:
            bool: upgrade pricing
        """
        for allocation in self.allocations:
            allocation[np.isclose(allocation, 0)] = 0
            allocation[np.isclose(allocation, 1)] = 1
        for permutation in permutations(range(len(self.allocations))):
            if np.array(
                [
                    (
                        self.allocations[permutation[i]]
                        <= self.allocations[permutation[i + 1]]
                    ).all()
                    for i in range(len(self.allocations) - 1)
                ]
            ).all():
                return True
        return False

    def is_interior(self):
        """Determines whether a solution is interior

        Returns:
            bool: interior
        """
        for allocation in self.allocations.T:
            if np.any(np.logical_or(allocation == 1, allocation == 0)):
                continue
            return True
        return False

    def save(self, filename):
        """visualizes and saves in an HTML file.

        Args:
            filename (string): filename to save html to
        """
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        points = ax.scatter(self._types[:, 0], self._types[:, 1], c="b")
        ax.scatter(self.allocations[:, 0], self.allocations[:, 1], c="r")
        arrow_tooltips = []
        point_labels = []
        for i in range(self.supp):
            point_labels.append(
                f"p:{self._probabilities[i]:.2f}\n virt{self.virtual[i,0]:.2f},{self.virtual[i,1]:.2f}\n myer{self.initial_virtual[i,0]:.2f},{self.initial_virtual[i,1]:.2f}\n type{self._types[i,0]:.2f},{self._types[i,1]:.2f}"
            )
            for j in range(self.supp):
                if i != j:
                    if self.lam[j][i] != 0:
                        arrow = ax.arrow(
                            self._types[i, 0],
                            self._types[i, 1],
                            self._types[j, 0] - self._types[i, 0],
                            self._types[j, 1] - self._types[i, 1],
                            head_width=0.01,
                            head_length=0.03,
                            width=0.005,
                            ec="black",
                            fc="black",
                            capstyle="round",
                            length_includes_head=True,
                        )
                        arrow_label = (
                            f"{self.lam[i][j]:.2f}, {self.initial_lam[i][j]:.2f}"
                        )
                        arrow_tooltips.append(
                            plugins.LineHTMLTooltip(
                                arrow, arrow_label, voffset=10, hoffset=10
                            )
                        )
        point_tooltip = plugins.PointHTMLTooltip(
            points, point_labels, voffset=10, hoffset=10
        )
        plugins.connect(fig, point_tooltip, *arrow_tooltips)
        with open(filename, "w") as file:
            mpld3.save_html(fig, file)


class InstanceRandomizer:
    """Class to generate random multi-goods monopolist instances"""

    def __init__(self, supp=6):
        """Constructor

        Args:
            supp (int, optional): Size of support of generated instances. Defaults to 6.
        """
        self.supp = supp

    def generate(self):
        """generates an instance

        Returns:
            Tuple[np.array, np.array]: arrays of types and probabilities of an instance
        """
        multipliers = np.array([np.random.uniform(1, 1.2) for i in range(self.supp)]).T
        second_multipliers = np.array(
            [np.random.uniform(1, 1.2) for i in range(self.supp)]
        ).T
        types = np.vstack(
            [
                np.cumprod(multipliers),
                np.cumprod(multipliers) * np.cumprod(second_multipliers),
            ]
        ).T
        types -= 0.5 * np.ones_like(types)
        probabilities = np.array([np.random.uniform() for i in range(self.supp)])
        probabilities /= probabilities.sum()
        return types, probabilities


def generate_upgrade_pricing_image(supp, num=20, default=False):
    """Top-level function to generate solved examples.

    Args:
        supp (int): Number of types
        num (int, optional): Number of images to generate. Defaults to 20.
        default (bool, optional): Choose a default upgrade pricing example. Defaults to False.
    """
    random = InstanceRandomizer(supp)
    for k in range(num):
        if default:
            types = np.array(
                [
                    [1300 / 25263.0, 4316 / 8421.0],
                    [2200 / 25263.0, 641 / 1203.0],
                    [5350 / 25263.0, 1103 / 1203.0],
                    [7850 / 25263.0, 1],
                ]
            )
            probabilities = [1 / 4.0, 3 / 20.0, 1 / 10.0, 1 / 2.0]
        else:
            types, probabilities = random.generate()
        monopolist = MGMProblem(types, probabilities)
        monopolist.solve()
        if monopolist.is_upgrade_pricing() and (not monopolist.is_grand_bundling()):
            monopolist.save(f"img_{k}.html")
            k += 1
