import numpy as np
import sympy as sp
from manipulatorKinematics import FK_Exponential
from lagrangianDynamic import LagrangeDynamic

def massPoint():
    link_count = 3
    l1, l2, l3 = sp.symbols('l1 l2 l3')  # Link lengths
    t1, t2, t3 = sp.symbols('t1 t2 t3')  # Joint angles
    thetas = [t1, t2, t3]

    M1 = [[1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    M2 = [[1, 0, 0, l2],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    M3 = [[1, 0, 0, l2 + l3],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]

    M = [M1, M2, M3]
    
    arr_w = [[0, 0, 1],
            [0, -1, 0],
            [0, -1, 0]]

    arr_q = [[0, 0, 0],
            [0, 0, l1],
            [l2, 0, l1]]
    
    masses = sp.symarray('m', link_count + 1)[1:]  # Masses of the links
    gravity = sp.symbols('g')  # Acceleration due to gravity
    
    fk = FK_Exponential(M, arr_w, arr_q, thetas)

    fk.set_thetas(thetas)

    dynamic_model = LagrangeDynamic(thetas, fk, None,  masses, gravity) # type: ignore

    # print("Torques required at each joint:")
    # tau = dynamic_model.torques
    # print(sp.simplify(tau))

    print ("Mass point: Computing D(q), C(q, qdot), and G(q) matrices...")
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
    l1, l2, l3 = sp.symbols('l1 l2 l3')  # Link lengths
    t1, t2, t3 = sp.symbols('t1 t2 t3')  # Joint angles
    lc1, lc2, lc3 = sp.symbols('lc1 lc2 lc3')  # Center of mass distances
    thetas = [t1, t2, t3]

    l1, l2, l3 = sp.symbols('l1 l2 l3')  # Link lengths
    t1, t2, t3 = sp.symbols('t1 t2 t3')  # Joint angles
    thetas = [t1, t2, t3]

    M1 = [[1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    M2 = [[1, 0, 0, l2],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    M3 = [[1, 0, 0, l2 + l3],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]

    M = [M1, M2, M3]

    Mc1 = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, lc1],
            [0, 0, 0, 1]]
    
    Mc2 = [[1, 0, 0, lc2],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    Mc3 = [[1, 0, 0, l2 + lc3],
            [0, 0, -1, 0],
            [0, 1, 0, l1],
            [0, 0, 0, 1]]
    
    Mc = [Mc1, Mc2, Mc3]
    
    arr_w = [[0, 0, 1],
            [0, -1, 0],
            [0, -1, 0]]

    arr_q = [[0, 0, 0],
            [0, 0, l1],
            [l2, 0, l1]]
    

    masses = sp.symarray('m', link_count + 1)[1:]  # Masses of the links
    gravity = sp.symbols('g')  # Acceleration due to gravity
    inertia_array = []
    for i in range(link_count):
        inertia = [
            sp.Matrix([[sp.symbols(f'Ixx{i+1}'), sp.symbols(f'Ixy{i+1}'), sp.symbols(f'Ixz{i+1}')],
                    [sp.symbols(f'Ixy{i+1}'), sp.symbols(f'Iyy{i+1}'), sp.symbols(f'Iyz{i+1}')],
                    [sp.symbols(f'Ixz{i+1}'), sp.symbols(f'Iyz{i+1}'), sp.symbols(f'Izz{i+1}')]])
        ]
        inertia_array.append(inertia)
    
    fkMc = FK_Exponential(Mc, arr_w, arr_q, thetas)
    fkM = FK_Exponential(M, arr_w, arr_q, thetas)

    dynamic_model = LagrangeDynamic(thetas, fkM, fkMc, masses, gravity)

    print ("Center mass: Computing D(q), C(q, qdot), and G(q) matrices...")
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


if __name__ == "__main__":
    massPoint()
    centerMass()