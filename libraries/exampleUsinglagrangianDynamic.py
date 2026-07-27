import numpy as np
import sympy as sp
from manipulatorKinematics import FK_Exponential
from libraries.bak.lagrangianDynamic import LagrangeDynamic
from lagrangianDynamicReal import LagrangeDynamicReal


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
    q = [t1, t2, t3]

    qddot = sp.symarray("qddot", link_count + 1)[1:]  # Joint accelerations
    qdot = sp.symarray("qdot", link_count + 1)[1:]  # Joint velocities


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

    fkMc = FK_Exponential(Mc, arr_w, arr_q, q)
    fkM = FK_Exponential(M, arr_w, arr_q, q)

    dynamic_model = LagrangeDynamic(fkM, fkMc, masses, gravity, inertia_array)  # type: ignore

    dynamic_model.get_DCG_matrices(q, qdot, qddot)

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

    dynamic_model = LagrangeDynamic(thetas, fkM, fkMc, masses, gravity, inertia_array)  # type: ignore

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



def ur5_number(q, qdot, qddot):
    """
    UR5e configuration using numerical geometry, COM locations,
    masses, and gravity from the Webots UR5e model.
    """
    link_count = 6
    joint_count = 6

    # Joint variables remain symbolic
    # t1, t2, t3, t4, t5, t6 = sp.symbols("t1 t2 t3 t4 t5 t6", real=True)
    # thetas = [t1, t2, t3, t4, t5, t6]

    # qdot = sp.Matrix(sp.symbols("qdot_1:7", real=True))

    # qddot = sp.Matrix(sp.symbols("qddot_1:7", real=True))

    # -----------------------------------------------------
    # Numerical UR5e parameters, in meters
    # Rational values keep symbolic calculations exact
    # -----------------------------------------------------

    gravity = sp.Rational(981, 100)  # 9.81 m/s^2

    d1 = sp.Rational(163, 1000)  # 0.163
    d2 = sp.Rational(138, 1000)  # 0.138
    a2 = sp.Rational(425, 1000)  # 0.425
    d3 = sp.Rational(131, 1000)  # 0.131
    a3 = sp.Rational(392, 1000)  # 0.392
    d4 = sp.Rational(127, 1000)  # 0.127
    d5 = sp.Rational(100, 1000)  # 0.100
    d6 = sp.Rational(100, 1000)  # 0.100 tool offset

    # COM offsets in each link's local frame
    c1y = -sp.Rational(2561, 100000)  # -0.02561
    c1z = sp.Rational(193, 100000)  #  0.00193

    c2z = sp.Rational(2125, 10000)  # 0.2125
    c3z = sp.Rational(15, 100)  # 0.150
    c4y = sp.Rational(127, 1000)  # 0.127
    c5z = sp.Rational(1, 10)  # 0.100
    c6y = sp.Rational(71, 1000)  # 0.071

    # -----------------------------------------------------
    # Link-frame home transformations
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # COM-frame home transformations
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Space-frame joint axes at the home configuration
    # -----------------------------------------------------

    arr_w = [
        [0, 0, 1],  # shoulder pan
        [0, 1, 0],  # shoulder lift
        [0, 1, 0],  # elbow
        [0, 1, 0],  # wrist 1
        [0, 0, -1],  # wrist 2
        [0, 1, 0],  # wrist 3
    ]

    # Points on joint axes in the base frame
    arr_q = [
        [0, 0, d1],
        [0, d2, d1],
        [a2, d2 - d3, d1],
        [a2 + a3, d2 - d3, d1],
        [a2 + a3, d2 - d3 + d4, d1],
        [a2 + a3, d2 - d3 + d4, d1 - d5],
    ]

    # -----------------------------------------------------
    # Numerical link masses, kg
    # -----------------------------------------------------

    masses = [
        sp.Rational(3761, 1000),  # 3.761 shoulder
        sp.Rational(8058, 1000),  # 8.058 upper arm
        sp.Rational(2846, 1000),  # 2.846 forearm
        sp.Rational(137, 100),  # 1.370 wrist 1
        sp.Rational(13, 10),  # 1.300 wrist 2
        sp.Rational(365, 1000),  # 0.365 wrist 3
    ]

    # -----------------------------------------------------
    # Symbolic inertia tensors
    #
    # Numerical inertia matrices are not explicitly stored
    # in the supplied Webots/URDF configuration.
    # Inerital diagnal is 0 0 0.0002 for the 6th joint, all others are 0 0 0
    # -----------------------------------------------------

    inertia_array = []
    inertia_array = [sp.zeros(3, 3) for _ in range(link_count)]

    # Sixth link inertia:
    # diag(Ixx6, Iyy6, Izz6) = diag(0, 0, 0.0002)
    inertia_array[5] = sp.diag(0, 0, sp.Rational(2, 10000))

    fkMc = FK_Exponential(Mc, arr_w, arr_q, q)

    fkM = FK_Exponential(M, arr_w, arr_q, q)

    # Include inertia_array if your constructor supports
    # rotational kinetic energy.
    dynamic_model = LagrangeDynamicReal(fkM, fkMc, masses, gravity, inertia_array)  # type: ignore

    print("Center mass: Computing D(q), C(q, qdot), " "and G(q) matrices...")

    # D, C, G = dynamic_model.get_DCG_matrices(q, qdot, qddot)  # type: ignore

    # print("D(q) matrix:")
    # sp.pprint(sp.trigsimp(D))

    # print("C(q, qdot) matrix:")
    # sp.pprint(sp.trigsimp(C))

    # print("G(q) matrix:")
    # sp.pprint(sp.trigsimp(G))

    tau = dynamic_model.get_torque(q, qdot, qddot)  # type: ignore

    print("tau")
    sp.pprint(tau)

    # with open("UR5_dcg_matrices_center_mass.tex", "w", encoding="utf-8") as file:
    #     file.write("\\section{D(q) matrix}\n")
    #     file.write(sp.latex(sp.trigsimp(D)))

    #     file.write("\n\n\\section{C(q, qdot) matrix}\n")
    #     file.write(sp.latex(sp.trigsimp(C)))

    #     file.write("\n\n\\section{G(q) matrix}\n")
    #     file.write(sp.latex(sp.trigsimp(G)))


if __name__ == "__main__":
    # ur5()
    # 3.200000000000000067e-02,2.039159568398015460e-06,-1.570794967355184202e+00,1.570796326794896558e+00,-1.570797686234608914e+00,-1.570796326794896558e+00,1.359439712265343569e-06,1.905567097195889350e-04,1.270378064797259747e-04,0.000000000000000000e+00,-1.270378064797259205e-04,0.000000000000000000e+00,1.270378064797259476e-04,1.183308070621843545e-02,7.888720470812292038e-03,0.000000000000000000e+00,-7.888720470812288568e-03,0.000000000000000000e+00,7.888720470812290303e-03,8.169450513918836282e-03,-2.025229716954466497e+01,-2.025879287722258937e+01,-1.824640082553683929e+00,3.323350884520274496e-04,-7.888720470812288812e-07

    q = [	0.0000,	-1.5708,	1.5708,	-1.5708,	-1.5708	,0.0000]
    qdot = [0.0000,	0.0000,	0.0000,	-0.0000,	0.0000,	0.0000]
    qddot = [	0.0000,	0.0000,	0.0000,	-0.0000	,0.0000, 0.0000]
    ur5_number(q, qdot, qddot)
    # massPoint()
    # centerMass()
