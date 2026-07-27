%% 3-DOF Rz-Ry-Ry Robot
% Newton-Euler inverse dynamics
%
% Joint 1 rotates about +z
% Joint 2 rotates about -y
% Joint 3 rotates about -y
%
% Robot geometry:
%   Link 1 extends vertically along z
%   Link 2 extends along its local x-axis
%   Link 3 extends along its local x-axis

clear;
clc;

%% ============================================================
% 1. Symbolic variables
% =============================================================

syms theta1 theta2 theta3 real
syms dtheta1 dtheta2 dtheta3 real
syms ddtheta1 ddtheta2 ddtheta3 real

syms l1 l2 l3 real
syms lc1 lc2 lc3 real

syms m1 m2 m3 real
syms g real

q = [
    theta1;
    theta2;
    theta3
];

qdot = [
    dtheta1;
    dtheta2;
    dtheta3
];

qddot = [
    ddtheta1;
    ddtheta2;
    ddtheta3
];

%% ============================================================
% 2. Link inertia matrices
% =============================================================

syms Ixx1 Iyy1 Izz1 Ixy1 Ixz1 Iyz1 real
syms Ixx2 Iyy2 Izz2 Ixy2 Ixz2 Iyz2 real
syms Ixx3 Iyy3 Izz3 Ixy3 Ixz3 Iyz3 real

% Inertia matrices expressed in each link body frame

I1_body = [
     Ixx1, -Ixy1, -Ixz1;
    -Ixy1,  Iyy1, -Iyz1;
    -Ixz1, -Iyz1,  Izz1
];

I2_body = [
     Ixx2, -Ixy2, -Ixz2;
    -Ixy2,  Iyy2, -Iyz2;
    -Ixz2, -Iyz2,  Izz2
];

I3_body = [
     Ixx3, -Ixy3, -Ixz3;
    -Ixy3,  Iyy3, -Iyz3;
    -Ixz3, -Iyz3,  Izz3
];

%% ============================================================
% 3. Rotation matrices
% =============================================================

R01 = rotZ(theta1);

% Your previous Ry matrix:
%
% [ cos(theta)   0  -sin(theta)
%       0        1       0
%   sin(theta)   0   cos(theta) ]
%
% This is a rotation about the negative y-axis.

R12 = rotMinusY(theta2);
R23 = rotMinusY(theta3);

% Link orientations relative to world frame

R02 = R01 * R12;
R03 = R02 * R23;

%% ============================================================
% 4. Joint axes expressed in world frame
% =============================================================

z_axis = [
    0;
    0;
    1
];

minus_y_axis = [
     0;
    -1;
     0
];

% Joint 1 axis

a1 = z_axis;

% Joint 2 axis rotated by link 1 orientation

a2 = R01 * minus_y_axis;

% Joint 3 axis rotated by links 1 and 2

a3 = R02 * minus_y_axis;

% a1 = simplify(a1);
% a2 = simplify(a2);
% a3 = simplify(a3);

disp('Joint axes in world frame:');

disp('a1 = ');
disp(a1);

disp('a2 = ');
disp(a2);

disp('a3 = ');
disp(a3);

%% ============================================================
% 5. Joint-origin positions
% =============================================================

% Joint 1 origin

p1 = [
    0;
    0;
    0
];

% Joint 2 is at the top of link 1

p2 = p1 + R01 * [
    0;
    0;
    l1
];

% Joint 3 is at the end of link 2

p3 = p2 + R02 * [
    l2;
    0;
    0
];

%% ============================================================
% 6. Center-of-mass positions
% =============================================================

% Link 1 COM

pc1 = p1 + R01 * [
    0;
    0;
    lc1
];

% Link 2 COM

pc2 = p2 + R02 * [
    lc2;
    0;
    0
];

% Link 3 COM

pc3 = p3 + R03 * [
    lc3;
    0;
    0
];

% p1  = simplify(p1);
% p2  = simplify(p2);
% p3  = simplify(p3);
% 
% pc1 = simplify(pc1);
% pc2 = simplify(pc2);
% pc3 = simplify(pc3);

disp('Center-of-mass positions:');

disp('pc1 = ');
disp(pc1);

disp('pc2 = ');
disp(pc2);

disp('pc3 = ');
disp(pc3);

%% ============================================================
% 7. Newton-Euler forward recursion
% =============================================================
%
% Calculate:
%   angular velocities
%   angular accelerations
%   joint-origin accelerations
%   center-of-mass accelerations

% Stationary base

omega0 = [
    0;
    0;
    0
];

alpha0 = [
    0;
    0;
    0
];

accel0 = [
    0;
    0;
    0
];

%% Link 1 angular velocity

omega1 = omega0 + a1*dtheta1;

%% Link 1 angular acceleration

alpha1 = alpha0 ...
       + a1*ddtheta1 ...
       + cross(omega0, a1*dtheta1);

%% Link 1 origin acceleration

accel1 = accel0;

%% Link 1 center-of-mass acceleration

r1c1 = pc1 - p1;

accel_c1 = accel1 ...
         + cross(alpha1, r1c1) ...
         + cross(omega1, cross(omega1, r1c1));

%% Link 2 angular velocity

omega2 = omega1 + a2*dtheta2;

%% Link 2 angular acceleration

alpha2 = alpha1 ...
       + a2*ddtheta2 ...
       + cross(omega1, a2*dtheta2);

%% Link 2 origin acceleration

r12 = p2 - p1;

accel2 = accel1 ...
       + cross(alpha1, r12) ...
       + cross(omega1, cross(omega1, r12));

%% Link 2 center-of-mass acceleration

r2c2 = pc2 - p2;

accel_c2 = accel2 ...
         + cross(alpha2, r2c2) ...
         + cross(omega2, cross(omega2, r2c2));

%% Link 3 angular velocity

omega3 = omega2 + a3*dtheta3;

%% Link 3 angular acceleration

alpha3 = alpha2 ...
       + a3*ddtheta3 ...
       + cross(omega2, a3*dtheta3);

%% Link 3 origin acceleration

r23 = p3 - p2;

accel3 = accel2 ...
       + cross(alpha2, r23) ...
       + cross(omega2, cross(omega2, r23));

%% Link 3 center-of-mass acceleration

r3c3 = pc3 - p3;

accel_c3 = accel3 ...
         + cross(alpha3, r3c3) ...
         + cross(omega3, cross(omega3, r3c3));

%% Simplify forward-recursion results

omega1 = simplify(omega1);
omega2 = simplify(omega2);
omega3 = simplify(omega3);

alpha1 = simplify(alpha1);
alpha2 = simplify(alpha2);
alpha3 = simplify(alpha3);

accel_c1 = simplify(accel_c1);
accel_c2 = simplify(accel_c2);
accel_c3 = simplify(accel_c3);

disp('Angular velocities:');

disp('omega1 = ');
disp(omega1);

disp('omega2 = ');
disp(omega2);

disp('omega3 = ');
disp(omega3);

disp('Angular accelerations:');

disp('alpha1 = ');
disp(alpha1);

disp('alpha2 = ');
disp(alpha2);

disp('alpha3 = ');
disp(alpha3);

disp('Center-of-mass accelerations:');

disp('accel_c1 = ');
disp(accel_c1);

disp('accel_c2 = ');
disp(accel_c2);

disp('accel_c3 = ');
disp(accel_c3);

%% ============================================================
% 8. Rotate inertia matrices into the world frame
% =============================================================

I1_world = simplify(R01 * I1_body * transpose(R01));
I2_world = simplify(R02 * I2_body * transpose(R02));
I3_world = simplify(R03 * I3_body * transpose(R03));

%% ============================================================
% 9. Inertial forces and inertial moments
% =============================================================

% Gravity acceleration vector in world frame

gravity = [
     0;
     0;
    -g
];

% Required force:
%
% F = m(a_c - gravity)
%
% Since gravity = [0;0;-g], a static robot produces an upward
% balancing force.

F1 = m1 * (accel_c1 - gravity);
F2 = m2 * (accel_c2 - gravity);
F3 = m3 * (accel_c3 - gravity);

% Required rotational moment around each center of mass:
%
% N = I*alpha + omega x (I*omega)

N1 = I1_world*alpha1 ...
   + cross(omega1, I1_world*omega1);

N2 = I2_world*alpha2 ...
   + cross(omega2, I2_world*omega2);

N3 = I3_world*alpha3 ...
   + cross(omega3, I3_world*omega3);

F1 = simplify(F1);
F2 = simplify(F2);
F3 = simplify(F3);

N1 = simplify(N1);
N2 = simplify(N2);
N3 = simplify(N3);

%% ============================================================
% 10. Newton-Euler backward recursion
% =============================================================
%
% The backward recursion starts at link 3 and moves toward link 1.
%
% f_i = total force transmitted through joint i
% n_i = total moment about joint i

%% Link 3

f3 = F3;

n3 = N3 ...
   + cross(pc3 - p3, F3);

tau3 = simplify(transpose(a3) * n3);

%% Link 2

f2 = F2 + f3;

n2 = N2 ...
   + n3 ...
   + cross(pc2 - p2, F2) ...
   + cross(p3 - p2, f3);

tau2 = simplify(transpose(a2) * n2);

%% Link 1

f1 = F1 + f2;

n1 = N1 ...
   + n2 ...
   + cross(pc1 - p1, F1) ...
   + cross(p2 - p1, f2);

tau1 = simplify(transpose(a1) * n1);

%% Joint torque vector

tau = [
    tau1;
    tau2;
    tau3
];

tau = simplify(tau, 'Steps', 50);

disp('Newton-Euler joint torque vector:');
disp('tau = ');
disp(tau);

%% ============================================================
% 11. Extract inertia matrix D(q)
% =============================================================
%
% Since:
%
% tau = D(q)qddot + nonlinear terms
%
% differentiate tau with respect to qddot.

D = jacobian(tau, qddot);

D = simplify(D, 'Steps', 50);

disp('Inertia matrix D(q):');
disp(D);

%% ============================================================
% 12. Extract gravity vector G(q)
% =============================================================
%
% Set all velocities and accelerations equal to zero.

zero_velocity = {
    dtheta1, dtheta2, dtheta3, ...
    ddtheta1, ddtheta2, ddtheta3
};

zero_values = {
    0, 0, 0, ...
    0, 0, 0
};

G = subs(tau, zero_velocity, zero_values);

G = simplify(G, 'Steps', 50);

disp('Gravity vector G(q):');
disp(G);

%% ============================================================
% 13. Calculate Coriolis matrix C(q,qdot)
% =============================================================
%
% Christoffel-symbol definition:
%
% C_ij = sum_k 1/2[
%           dD_ij/dq_k
%         + dD_ik/dq_j
%         - dD_jk/dq_i
%       ] qdot_k

n = length(q);

C = sym(zeros(n,n));

for i = 1:n
    for j = 1:n

        Cij = sym(0);

        for k = 1:n

            christoffel = sym(1)/2 * ( ...
                diff(D(i,j), q(k)) ...
              + diff(D(i,k), q(j)) ...
              - diff(D(j,k), q(i)) ...
            );

            Cij = Cij + christoffel*qdot(k);
        end

        C(i,j) = simplify(Cij);
    end
end

C = simplify(C, 'Steps', 50);

disp('Coriolis and centrifugal matrix C(q,qdot):');
disp(C);

%% ============================================================
% 14. Verify the standard dynamic equation
% =============================================================

tau_standard = D*qddot + C*qdot + G;

verification = simplify(tau - tau_standard, 'Steps', 100);

disp('Verification: tau - (D*qddot + C*qdot + G)');
disp(verification);

% Correct result:
%
% [0
%  0
%  0]

%% ============================================================
% 15. Optional checks
% =============================================================

disp('Check whether D is symmetric:');

D_symmetry_check = simplify(D - transpose(D));

disp(D_symmetry_check);

% Correct result:
%
% zeros(3,3)

%% ============================================================
% 16. Save results
% =============================================================

save('newton_euler_results.mat', ...
    'tau', ...
    'D', ...
    'C', ...
    'G', ...
    'omega1', ...
    'omega2', ...
    'omega3', ...
    'alpha1', ...
    'alpha2', ...
    'alpha3', ...
    'accel_c1', ...
    'accel_c2', ...
    'accel_c3');

disp('Results saved to newton_euler_results.mat');

%% ============================================================
% Local rotation functions
% =============================================================

function R = rotZ(theta)

    R = [
        cos(theta), -sin(theta), 0;
        sin(theta),  cos(theta), 0;
        0,           0,          1
    ];

end

function R = rotMinusY(theta)

    R = [
         cos(theta), 0, -sin(theta);
         0,          1,  0;
         sin(theta), 0,  cos(theta)
    ];

end