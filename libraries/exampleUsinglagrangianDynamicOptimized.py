import numpy as np
import sympy as sp
from manipulatorKinematicsOptimized import FK_Exponential
from lagrangianDynamicOptimized import LagrangeDynamic


def massPoint():
    link_count = 3
    l1, l2, l3 = sp.symbols("l1 l2 l3")  # Link lengths
    t1, t2, t3 = sp.symbols("t1 t2 t3")  # Joint angles
    thetas = [t1, t2, t3]

    M1 = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M2 = [[1, 0, 0, l2], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M3 = [[1, 0, 0, l2 + l3], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M = [M1, M2, M3]

    arr_w = [[0, 0, 1], [0, -1, 0], [0, -1, 0]]

    arr_q = [[0, 0, 0], [0, 0, l1], [l2, 0, l1]]

    masses = sp.symarray("m", link_count + 1)[1:]  # Masses of the links
    gravity = sp.symbols("g")  # Acceleration due to gravity

    fk = FK_Exponential(M, arr_w, arr_q, thetas)

    fk.set_thetas(thetas)

    dynamic_model = LagrangeDynamic(thetas, fk, None, masses, gravity)  # type: ignore

    # print("Torques required at each joint:")
    # tau = dynamic_model.torques
    # print(sp.simplify(tau))

    print("Mass point: Computing D(q), C(q, qdot), and G(q) matrices...")
    D, C, G = dynamic_model.DCG_matrices
    print("D(q) matrix:")
    sp.pprint(sp.trigsimp(D))
    print("C(q, qdot) matrix:")
    sp.pprint(sp.trigsimp(C))
    print("G(q) matrix:")
    sp.pprint(sp.trigsimp(G))

    # save all DCG as latex to a file
    with open("dcg_matrices_mass_point.tex", "w") as f:
        f.write("\\section{D(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(D)))
        f.write("\n\n\\section{C(q, qdot) matrix}\n")
        f.write(sp.latex(sp.trigsimp(C)))
        f.write("\n\n\\section{G(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(G)))


def centerMass():
    link_count = 3
    l1, l2, l3 = sp.symbols("l1 l2 l3")  # Link lengths
    t1, t2, t3 = sp.symbols("t1 t2 t3")  # Joint angles
    lc1, lc2, lc3 = sp.symbols("lc1 lc2 lc3")  # Center of mass distances
    thetas = [t1, t2, t3]

    l1, l2, l3 = sp.symbols("l1 l2 l3")  # Link lengths
    t1, t2, t3 = sp.symbols("t1 t2 t3")  # Joint angles
    thetas = [t1, t2, t3]

    M1 = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M2 = [[1, 0, 0, l2], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M3 = [[1, 0, 0, l2 + l3], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    M = [M1, M2, M3]

    Mc1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, lc1], [0, 0, 0, 1]]

    Mc2 = [[1, 0, 0, lc2], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    Mc3 = [[1, 0, 0, l2 + lc3], [0, 0, -1, 0], [0, 1, 0, l1], [0, 0, 0, 1]]

    Mc = [Mc1, Mc2, Mc3]

    arr_w = [[0, 0, 1], [0, -1, 0], [0, -1, 0]]

    arr_q = [[0, 0, 0], [0, 0, l1], [l2, 0, l1]]

    masses = sp.symarray("m", link_count + 1)[1:]  # Masses of the links
    gravity = sp.symbols("g")  # Acceleration due to gravity
    inertia_array = []
    for i in range(link_count):
        inertia = [
            sp.Matrix(
                [
                    [
                        sp.symbols(f"Ixx{i+1}"),
                        sp.symbols(f"Ixy{i+1}"),
                        sp.symbols(f"Ixz{i+1}"),
                    ],
                    [
                        sp.symbols(f"Ixy{i+1}"),
                        sp.symbols(f"Iyy{i+1}"),
                        sp.symbols(f"Iyz{i+1}"),
                    ],
                    [
                        sp.symbols(f"Ixz{i+1}"),
                        sp.symbols(f"Iyz{i+1}"),
                        sp.symbols(f"Izz{i+1}"),
                    ],
                ]
            )
        ]
        inertia_array.append(inertia)

    fkMc = FK_Exponential(Mc, arr_w, arr_q, thetas)
    fkM = FK_Exponential(M, arr_w, arr_q, thetas)

    dynamic_model = LagrangeDynamic(thetas, fkM, fkMc, masses, gravity)

    print("Center mass: Computing D(q), C(q, qdot), and G(q) matrices...")
    D, C, G = dynamic_model.DCG_matrices
    print("D(q) matrix:")
    sp.pprint(sp.trigsimp(D))
    print("C(q, qdot) matrix:")
    sp.pprint(sp.trigsimp(C))
    print("G(q) matrix:")
    sp.pprint(sp.trigsimp(G))

    # save all DCG as latex to a file
    with open("dcg_matrices_center_mass.tex", "w") as f:
        f.write("\\section{D(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(D)))
        f.write("\n\n\\section{C(q, qdot) matrix}\n")
        f.write(sp.latex(sp.trigsimp(C)))
        f.write("\n\n\\section{G(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(G)))


def ur5():
    """
    Example usage for the UR5 robot. Based on the UR5e.urdf in protos\\UR5e.urdf
    """
    link_count = 6
    joint_count = 6

    # Joint angles
    t1, t2, t3, t4, t5, t6 = sp.symbols("t1 t2 t3 t4 t5 t6", real=True)

    thetas = [t1, t2, t3, t4, t5, t6]

    # Joint velocities and accelerations
    qdot = sp.Matrix(sp.symbols("qdot_1:7", real=True))

    qddot = sp.Matrix(sp.symbols("qddot_1:7", real=True))

    # Gravity
    gravity = sp.symbols("g", positive=True, real=True)

    # Vertical base/shoulder offset
    d1 = sp.symbols("d1", positive=True, real=True)

    # Shoulder lateral offset
    d2 = sp.symbols("d2", real=True)

    # Upper-arm length
    a2 = sp.symbols("a2", positive=True, real=True)

    # Elbow lateral offset
    d3 = sp.symbols("d3", real=True)

    # Forearm length
    a3 = sp.symbols("a3", positive=True, real=True)

    # Wrist offsets
    d4, d5, d6 = sp.symbols("d4 d5 d6", positive=True, real=True)

    c1y, c1z = sp.symbols("c1y c1z", real=True)

    c2z = sp.symbols("c2z", real=True)
    c3z = sp.symbols("c3z", real=True)

    c4y = sp.symbols("c4y", real=True)
    c5z = sp.symbols("c5z", real=True)
    c6y = sp.symbols("c6y", real=True)

    M1 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]])

    M2 = sp.Matrix([[0, 0, 1, 0], [0, 1, 0, d2], [-1, 0, 0, d1], [0, 0, 0, 1]])

    M3 = sp.Matrix([[0, 0, 1, a2], [0, 1, 0, d2 - d3], [-1, 0, 0, d1], [0, 0, 0, 1]])

    M4 = sp.Matrix(
        [[-1, 0, 0, a2 + a3], [0, 1, 0, d2 - d3], [0, 0, -1, d1], [0, 0, 0, 1]]
    )

    M5 = sp.Matrix(
        [[-1, 0, 0, a2 + a3], [0, 1, 0, d2 - d3 + d4], [0, 0, -1, d1], [0, 0, 0, 1]]
    )

    M6 = sp.Matrix(
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4],
            [0, 0, -1, d1 - d5],
            [0, 0, 0, 1],
        ]
    )

    M = [M1, M2, M3, M4, M5, M6]

    Mc1 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, c1y], [0, 0, 1, d1 + c1z], [0, 0, 0, 1]])

    Mc2 = sp.Matrix([[0, 0, 1, c2z], [0, 1, 0, d2], [-1, 0, 0, d1], [0, 0, 0, 1]])

    Mc3 = sp.Matrix(
        [[0, 0, 1, a2 + c3z], [0, 1, 0, d2 - d3], [-1, 0, 0, d1], [0, 0, 0, 1]]
    )

    Mc4 = sp.Matrix(
        [[-1, 0, 0, a2 + a3], [0, 1, 0, d2 - d3 + c4y], [0, 0, -1, d1], [0, 0, 0, 1]]
    )

    Mc5 = sp.Matrix(
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4],
            [0, 0, -1, d1 - c5z],
            [0, 0, 0, 1],
        ]
    )

    Mc6 = sp.Matrix(
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4 + c6y],
            [0, 0, -1, d1 - d5],
            [0, 0, 0, 1],
        ]
    )

    Mc = [Mc1, Mc2, Mc3, Mc4, Mc5, Mc6]

    arr_w = [
        [0, 0, 1],  # shoulder pan
        [0, 1, 0],  # shoulder lift
        [0, 1, 0],  # elbow
        [0, 1, 0],  # wrist 1
        [0, 0, -1],  # wrist 2
        [0, 1, 0],  # wrist 3
    ]

    arr_q = [
        [0, 0, d1],
        [0, d2, d1],
        [a2, d2 - d3, d1],
        [a2 + a3, d2 - d3, d1],
        [a2 + a3, d2 - d3 + d4, d1],
        [a2 + a3, d2 - d3 + d4, d1 - d5],
    ]

    m1, m2, m3, m4, m5, m6 = sp.symbols("m1 m2 m3 m4 m5 m6", positive=True, real=True)

    masses = [m1, m2, m3, m4, m5, m6]

    inertia_array = []

    for i in range(1, link_count + 1):
        Ixx = sp.symbols(f"Ixx{i}", real=True)
        Iyy = sp.symbols(f"Iyy{i}", real=True)
        Izz = sp.symbols(f"Izz{i}", real=True)

        Ixy = sp.symbols(f"Ixy{i}", real=True)
        Ixz = sp.symbols(f"Ixz{i}", real=True)
        Iyz = sp.symbols(f"Iyz{i}", real=True)

        I = sp.Matrix(
            [
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz],
            ]
        )

        inertia_array.append(I)

    fkMc = FK_Exponential(Mc, arr_w, arr_q, thetas)
    fkM = FK_Exponential(M, arr_w, arr_q, thetas)

    dynamic_model = LagrangeDynamic(thetas, fkM, fkMc, masses, gravity)

    # dynamic_model.DCG_matrices

    print("Center mass: Computing D(q), C(q, qdot), and G(q) matrices...")
    D, C, G = dynamic_model.DCG_matrices
    print("D(q) matrix:")
    sp.pprint(sp.trigsimp(D))
    print("C(q, qdot) matrix:")
    sp.pprint(sp.trigsimp(C))
    print("G(q) matrix:")
    sp.pprint(sp.trigsimp(G))

    # save all DCG as latex to a file
    with open("UR5_dcg_matrices_center_mass.tex", "w") as f:
        f.write("\\section{D(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(D)))
        f.write("\n\n\\section{C(q, qdot) matrix}\n")
        f.write(sp.latex(sp.trigsimp(C)))
        f.write("\n\n\\section{G(q) matrix}\n")
        f.write(sp.latex(sp.trigsimp(G)))


if __name__ == "__main__":
    # Start with the smaller regression case. The fully symbolic UR5 model is
    # substantially more expensive, even with the optimized dynamics builder.
    # massPoint()
    ur5()
    # centerMass()
