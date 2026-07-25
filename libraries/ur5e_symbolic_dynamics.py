"""Step-by-step symbolic dynamics model of the UR5e.

This script intentionally keeps the computations explicit so that each stage
of the derivation can be inspected individually.

Install SymPy:
    python3 -m pip install sympy

Run:
    python3 ur5e_symbolic_dynamics.py
"""

import sympy as sp


def exact(value):
    """Convert a decimal value to an exact symbolic number."""
    return sp.Rational(str(value))


def simplify(expression):
    """Simplify a scalar or every element of a matrix."""
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(sp.simplify)
    return sp.simplify(expression)


def dh_transform(theta, a, d, alpha):
    """Create a standard Denavit-Hartenberg transformation matrix."""
    return sp.Matrix([
        [
            sp.cos(theta),
            -sp.sin(theta) * sp.cos(alpha),
            sp.sin(theta) * sp.sin(alpha),
            a * sp.cos(theta),
        ],
        [
            sp.sin(theta),
            sp.cos(theta) * sp.cos(alpha),
            -sp.cos(theta) * sp.sin(alpha),
            a * sp.sin(theta),
        ],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1],
    ])


def derive_dynamics():
    # ==================================================================
    # 1. SYMBOLIC JOINT VARIABLES
    # ==================================================================
    theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols(
        "theta1 theta2 theta3 theta4 theta5 theta6", real=True
    )

    theta1_dot, theta2_dot, theta3_dot, theta4_dot, theta5_dot, theta6_dot = (
        sp.symbols(
            "theta1_dot theta2_dot theta3_dot "
            "theta4_dot theta5_dot theta6_dot",
            real=True,
        )
    )

    (
        theta1_ddot,
        theta2_ddot,
        theta3_ddot,
        theta4_ddot,
        theta5_ddot,
        theta6_ddot,
    ) = sp.symbols(
        "theta1_ddot theta2_ddot theta3_ddot "
        "theta4_ddot theta5_ddot theta6_ddot",
        real=True,
    )

    q = sp.Matrix([theta1, theta2, theta3, theta4, theta5, theta6])
    q_dot = sp.Matrix([
        theta1_dot,
        theta2_dot,
        theta3_dot,
        theta4_dot,
        theta5_dot,
        theta6_dot,
    ])
    q_ddot = sp.Matrix([
        theta1_ddot,
        theta2_ddot,
        theta3_ddot,
        theta4_ddot,
        theta5_ddot,
        theta6_ddot,
    ])

    # ==================================================================
    # 2. STANDARD DH TRANSFORMATIONS
    # ==================================================================
    a1, a2, a3, a4, a5, a6 = map(
        exact, [0, -0.425, -0.3922, 0, 0, 0]
    )
    d1, d2, d3, d4, d5, d6 = map(
        exact, [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    )
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = (
        sp.pi / 2,
        0,
        0,
        sp.pi / 2,
        -sp.pi / 2,
        0,
    )

    T1 = simplify(dh_transform(theta1, a1, d1, alpha1))
    T2 = simplify(dh_transform(theta2, a2, d2, alpha2))
    T3 = simplify(dh_transform(theta3, a3, d3, alpha3))
    T4 = simplify(dh_transform(theta4, a4, d4, alpha4))
    T5 = simplify(dh_transform(theta5, a5, d5, alpha5))
    T6 = simplify(dh_transform(theta6, a6, d6, alpha6))

    T01 = T1
    T02 = simplify(T01 * T2)
    T03 = simplify(T02 * T3)
    T04 = simplify(T03 * T4)
    T05 = simplify(T04 * T5)
    T06 = simplify(T05 * T6)

    # ==================================================================
    # 3. CENTER-OF-MASS POSITIONS
    # ==================================================================
    # Each vector is expressed in its corresponding link frame.
    rc1 = sp.Matrix([0, exact(-0.02561), exact(0.00193), 1])
    rc2 = sp.Matrix([exact(0.2125), 0, exact(0.11336), 1])
    rc3 = sp.Matrix([exact(0.15), 0, exact(0.0265), 1])
    rc4 = sp.Matrix([0, exact(-0.0018), exact(0.01634), 1])
    rc5 = sp.Matrix([0, exact(0.0018), exact(0.01634), 1])
    rc6 = sp.Matrix([0, 0, exact(-0.001159), 1])

    pc1_h = simplify(T01 * rc1)
    pc2_h = simplify(T02 * rc2)
    pc3_h = simplify(T03 * rc3)
    pc4_h = simplify(T04 * rc4)
    pc5_h = simplify(T05 * rc5)
    pc6_h = simplify(T06 * rc6)

    pc1 = pc1_h[:3, :]
    pc2 = pc2_h[:3, :]
    pc3 = pc3_h[:3, :]
    pc4 = pc4_h[:3, :]
    pc5 = pc5_h[:3, :]
    pc6 = pc6_h[:3, :]

    # ==================================================================
    # 4. LINEAR JACOBIANS AND CENTER-OF-MASS VELOCITIES
    # ==================================================================
    Jv1 = simplify(pc1.jacobian(q))
    Jv2 = simplify(pc2.jacobian(q))
    Jv3 = simplify(pc3.jacobian(q))
    Jv4 = simplify(pc4.jacobian(q))
    Jv5 = simplify(pc5.jacobian(q))
    Jv6 = simplify(pc6.jacobian(q))

    vc1 = simplify(Jv1 * q_dot)
    vc2 = simplify(Jv2 * q_dot)
    vc3 = simplify(Jv3 * q_dot)
    vc4 = simplify(Jv4 * q_dot)
    vc5 = simplify(Jv5 * q_dot)
    vc6 = simplify(Jv6 * q_dot)

    # ==================================================================
    # 5. ANGULAR JACOBIANS AND ANGULAR VELOCITIES
    # ==================================================================
    z0 = sp.Matrix([0, 0, 1])
    z1 = simplify(T01[:3, 2])
    z2 = simplify(T02[:3, 2])
    z3 = simplify(T03[:3, 2])
    z4 = simplify(T04[:3, 2])
    z5 = simplify(T05[:3, 2])
    zero3 = sp.zeros(3, 1)

    Jw1 = simplify(sp.Matrix.hstack(z0, zero3, zero3, zero3, zero3, zero3))
    Jw2 = simplify(sp.Matrix.hstack(z0, z1, zero3, zero3, zero3, zero3))
    Jw3 = simplify(sp.Matrix.hstack(z0, z1, z2, zero3, zero3, zero3))
    Jw4 = simplify(sp.Matrix.hstack(z0, z1, z2, z3, zero3, zero3))
    Jw5 = simplify(sp.Matrix.hstack(z0, z1, z2, z3, z4, zero3))
    Jw6 = simplify(sp.Matrix.hstack(z0, z1, z2, z3, z4, z5))

    omega1 = simplify(Jw1 * q_dot)
    omega2 = simplify(Jw2 * q_dot)
    omega3 = simplify(Jw3 * q_dot)
    omega4 = simplify(Jw4 * q_dot)
    omega5 = simplify(Jw5 * q_dot)
    omega6 = simplify(Jw6 * q_dot)

    # ==================================================================
    # 6. MASSES, GRAVITY, AND INERTIA MATRICES
    # ==================================================================
    m1 = exact(3.761)
    m2 = exact(8.058)
    m3 = exact(2.846)
    m4 = exact(1.37)
    m5 = exact(1.3)
    m6 = exact(0.365)

    g = exact(9.81)
    g0 = sp.Matrix([0, 0, -g])

    I1 = sp.zeros(3)
    I2 = sp.zeros(3)
    I3 = sp.zeros(3)
    I4 = sp.zeros(3)
    I5 = sp.zeros(3)
    I6 = sp.diag(0, 0, exact(0.0002))

    R01 = T01[:3, :3]
    R02 = T02[:3, :3]
    R03 = T03[:3, :3]
    R04 = T04[:3, :3]
    R05 = T05[:3, :3]
    R06 = T06[:3, :3]

    I01 = simplify(R01 * I1 * R01.T)
    I02 = simplify(R02 * I2 * R02.T)
    I03 = simplify(R03 * I3 * R03.T)
    I04 = simplify(R04 * I4 * R04.T)
    I05 = simplify(R05 * I5 * R05.T)
    I06 = simplify(R06 * I6 * R06.T)

    # ==================================================================
    # 7. KINETIC ENERGY
    # ==================================================================
    # Linear kinetic energy
    Kv1 = simplify(sp.Rational(1, 2) * m1 * (vc1.T * vc1)[0])
    Kv2 = simplify(sp.Rational(1, 2) * m2 * (vc2.T * vc2)[0])
    Kv3 = simplify(sp.Rational(1, 2) * m3 * (vc3.T * vc3)[0])
    Kv4 = simplify(sp.Rational(1, 2) * m4 * (vc4.T * vc4)[0])
    Kv5 = simplify(sp.Rational(1, 2) * m5 * (vc5.T * vc5)[0])
    Kv6 = simplify(sp.Rational(1, 2) * m6 * (vc6.T * vc6)[0])

    # Rotational kinetic energy
    Kw1 = simplify(sp.Rational(1, 2) * (omega1.T * I01 * omega1)[0])
    Kw2 = simplify(sp.Rational(1, 2) * (omega2.T * I02 * omega2)[0])
    Kw3 = simplify(sp.Rational(1, 2) * (omega3.T * I03 * omega3)[0])
    Kw4 = simplify(sp.Rational(1, 2) * (omega4.T * I04 * omega4)[0])
    Kw5 = simplify(sp.Rational(1, 2) * (omega5.T * I05 * omega5)[0])
    Kw6 = simplify(sp.Rational(1, 2) * (omega6.T * I06 * omega6)[0])

    # Total kinetic energy of each link
    K1 = simplify(Kv1 + Kw1)
    K2 = simplify(Kv2 + Kw2)
    K3 = simplify(Kv3 + Kw3)
    K4 = simplify(Kv4 + Kw4)
    K5 = simplify(Kv5 + Kw5)
    K6 = simplify(Kv6 + Kw6)

    K_total = simplify(K1 + K2 + K3 + K4 + K5 + K6)

    # ==================================================================
    # 8. POTENTIAL ENERGY
    # ==================================================================
    P1 = simplify(-m1 * (g0.T * pc1)[0])
    P2 = simplify(-m2 * (g0.T * pc2)[0])
    P3 = simplify(-m3 * (g0.T * pc3)[0])
    P4 = simplify(-m4 * (g0.T * pc4)[0])
    P5 = simplify(-m5 * (g0.T * pc5)[0])
    P6 = simplify(-m6 * (g0.T * pc6)[0])

    P_total = simplify(P1 + P2 + P3 + P4 + P5 + P6)

    # ==================================================================
    # 9. LAGRANGIAN AND EULER-LAGRANGE EQUATIONS
    # ==================================================================
    L = simplify(K_total - P_total)

    dL_dqdot = sp.Matrix([sp.diff(L, velocity) for velocity in q_dot])
    dL_dq = sp.Matrix([sp.diff(L, coordinate) for coordinate in q])

    d_dt_dL_dqdot = (
        dL_dqdot.jacobian(q) * q_dot
        + dL_dqdot.jacobian(q_dot) * q_ddot
    )

    tau = simplify(d_dt_dL_dqdot - dL_dq)

    # ==================================================================
    # 10. COMPACT DYNAMIC MODEL
    #     tau = M(q) q_ddot + C(q, q_dot) q_dot + G(q)
    # ==================================================================
    M = simplify(tau.jacobian(q_ddot))
    h = simplify(tau - M * q_ddot)

    zero_joint_velocities = {velocity: 0 for velocity in q_dot}
    G = simplify(h.subs(zero_joint_velocities))
    Cqdot = simplify(h - G)

    # Collect the outputs in groups without hiding the manual derivation.
    return {
        "joint_variables": {
            "q": q,
            "q_dot": q_dot,
            "q_ddot": q_ddot,
        },
        "transformations": {
            "T01": T01,
            "T02": T02,
            "T03": T03,
            "T04": T04,
            "T05": T05,
            "T06": T06,
        },
        "center_of_mass_positions": {
            "pc1": pc1,
            "pc2": pc2,
            "pc3": pc3,
            "pc4": pc4,
            "pc5": pc5,
            "pc6": pc6,
        },
        "linear_jacobians": {
            "Jv1": Jv1,
            "Jv2": Jv2,
            "Jv3": Jv3,
            "Jv4": Jv4,
            "Jv5": Jv5,
            "Jv6": Jv6,
        },
        "angular_jacobians": {
            "Jw1": Jw1,
            "Jw2": Jw2,
            "Jw3": Jw3,
            "Jw4": Jw4,
            "Jw5": Jw5,
            "Jw6": Jw6,
        },
        "energies": {
            "K1": K1,
            "K2": K2,
            "K3": K3,
            "K4": K4,
            "K5": K5,
            "K6": K6,
            "K_total": K_total,
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "P4": P4,
            "P5": P5,
            "P6": P6,
            "P_total": P_total,
            "L": L,
        },
        "dynamic_model": {
            "tau": tau,
            "M": M,
            "h": h,
            "G": G,
            "Cqdot": Cqdot,
        },
    }


if __name__ == "__main__":
    print("Deriving the symbolic UR5e model. This may take a while...")
    results = derive_dynamics()
    print("Derivation complete.")
    print("Final outputs: tau, M, h, G, and Cqdot")
