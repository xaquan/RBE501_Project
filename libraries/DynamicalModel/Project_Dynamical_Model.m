clear;
clc;

% Symbolic joint variables
syms theta1 theta2 theta3 theta4 theta5 theta6 real
syms theta1_dot theta2_dot theta3_dot theta4_dot theta5_dot theta6_dot real
syms theta1_ddot theta2_ddot theta3_ddot theta4_ddot ...
    theta5_ddot theta6_ddot real
syms box_mass real %mass defined parametrically such that it can be changed

q = [theta1, theta2, theta3, theta4, theta5, theta6];

% UR5e standard DH parameters
a = [sym(0), sym('-0.425'), sym('-0.3922'), sym(0), sym(0), sym(0)];

d = [sym('0.1625'), sym(0), sym(0), sym('0.1333'), sym('0.0997'),...
    sym('0.0996')];

%box(payload dimensions) - the white or black box in the werobot world
box_length = sym('0.1');
box_width = sym('0.1');
box_height = sym('0.1');


% Exact sine and cosine values of the UR5e DH twist angles
cos_alpha = sym([0, 1, 1, 0, 0, 1]);
sin_alpha = sym([1, 0, 0, 1, -1, 0]);

% Preallocate symbolic 4x4x6 array
T = sym(zeros(4, 4, 6));

for i = 1:6
    T(:, :, i) = [
        cos(q(i)), -sin(q(i))*cos_alpha(i),  sin(q(i))*sin_alpha(i), a(i)*cos(q(i));
        sin(q(i)),  cos(q(i))*cos_alpha(i), -cos(q(i))*sin_alpha(i), a(i)*sin(q(i));
        0,              sin_alpha(i),                cos_alpha(i),               d(i);
        0,              0,                            0,                           1
    ];

    T(:, :, i) = simplify(T(:, :, i));
end

%Transformation matrices
T01 = T(:,:,1);
T02 = T(:,:,1)*T(:,:,2);
T02 = simplify(T02, 'Steps', 10);
T03 = simplify(T(:,:,1)*T(:,:,2)*T(:,:,3));
T03 = simplify(T03, 'Steps', 10);
T04 = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4);
T04 = simplify(T04, 'Steps', 10);
T05 = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4)*T(:,:,5);
T05 = simplify(T05, 'Steps', 10);
T06 = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4)*T(:,:,5)*T(:,:,6);
T06 = simplify(T06, 'Steps', 10);

% Center-of-mass vectors from the UR5e table
rc1 = [sym(0); sym('-0.02561'); sym('0.00193'); sym(1)];
rc2 = [sym('0.2125'); sym(0); sym('0.11336'); sym(1)];
rc3 = [sym('0.15'); sym(0); sym('0.0265'); sym(1)];
rc4 = [sym(0); sym('-0.0018'); sym('0.01634'); sym(1)];
rc5 = [sym(0); sym('0.0018'); sym('0.01634'); sym(1)];
rc6 = [sym(0); sym(0); sym('-0.001159'); sym(1)];
rc_box = [sym(0); sym(0); sym('-0.05'); sym(1)];

%representing the CoMs in the base frame
pc1_h = simplify(T01*rc1);
pc2_h = simplify(T02*rc2);
pc3_h = simplify(T03*rc3);
pc4_h = simplify(T04*rc4);
pc5_h = simplify(T05*rc5);
pc6_h = simplify(T06*rc6);
pc_box_h = simplify(T06*rc_box);

pc1 = pc1_h(1:3);
pc2 = pc2_h(1:3);
pc3 = pc3_h(1:3);
pc4 = pc4_h(1:3);
pc5 = pc5_h(1:3);
pc6 = pc6_h(1:3);
pc_box = pc_box_h(1:3);

q_dot = [theta1_dot; theta2_dot; theta3_dot; theta4_dot; theta5_dot; ...
    theta6_dot];

q_ddot = [theta1_ddot; theta2_ddot; theta3_ddot; theta4_ddot;...
    theta5_ddot; theta6_ddot];

%Calculating Linear jacobians
Jv1 = simplify(jacobian(pc1, q), 'Steps', 10);
Jv2 = simplify(jacobian(pc2, q), 'Steps', 10);
Jv3 = simplify(jacobian(pc3, q), 'Steps', 10);
Jv4 = simplify(jacobian(pc4, q), 'Steps', 10);
Jv5 = simplify(jacobian(pc5, q), 'Steps', 10);
Jv6 = simplify(jacobian(pc6, q), 'Steps', 10);
Jv_box = simplify(jacobian(pc_box, q), 'Steps', 10);

vc1 = simplify(Jv1*q_dot, 'Steps', 10);
vc2 = simplify(Jv2*q_dot, 'Steps', 10);
vc3 = simplify(Jv3*q_dot, 'Steps', 10);
vc4 = simplify(Jv4*q_dot, 'Steps', 10);
vc5 = simplify(Jv5*q_dot, 'Steps', 10);
vc6 = simplify(Jv6*q_dot, 'Steps', 10);
%vc_box = simplify(Jv_box*q_dot, 'Steps', 10);

z0 = sym([0; 0; 1]);
z1 = simplify(T01(1:3,3), 'Steps', 10);
z2 = simplify(T02(1:3,3), 'Steps', 10);
z3 = simplify(T03(1:3,3), 'Steps', 10);
z4 = simplify(T04(1:3,3), 'Steps', 10);
z5 = simplify(T05(1:3,3), 'Steps', 10);

zero3 = sym(zeros(3,1));

Jw1 = [z0, zero3, zero3, zero3, zero3, zero3];
Jw2 = [z0, z1, zero3, zero3, zero3, zero3];
Jw3 = [z0, z1, z2, zero3, zero3, zero3];
Jw4 = [z0, z1, z2, z3, zero3, zero3];
Jw5 = [z0, z1, z2, z3, z4, zero3];
Jw6 = [z0, z1, z2, z3, z4, z5];
Jw_box = Jw6;

Jw1 = simplify(Jw1, 'Steps', 10);
Jw2 = simplify(Jw2, 'Steps', 10);
Jw3 = simplify(Jw3, 'Steps', 10);
Jw4 = simplify(Jw4, 'Steps', 10);
Jw5 = simplify(Jw5, 'Steps', 10);
Jw6 = simplify(Jw6, 'Steps', 10);

omega1 = simplify(Jw1*q_dot, 'Steps', 10);
omega2 = simplify(Jw2*q_dot, 'Steps', 10);
omega3 = simplify(Jw3*q_dot, 'Steps', 10);
omega4 = simplify(Jw4*q_dot, 'Steps', 10);
omega5 = simplify(Jw5*q_dot, 'Steps', 10);
omega6 = simplify(Jw6*q_dot, 'Steps', 10);
%omega_box = omega6;

%Link Masses and gravity
m1 = sym('3.761');
m2 = sym('8.058');
m3 = sym('2.846');
m4 = sym('1.37');
m5 = sym('1.3');
m6 = sym('0.365');
g = sym('9.81');

%Gravity vector
g0 = [0; 0; -g];

% Define the inertia matrices for each link
I1 = zeros(3,3);
I2 = zeros(3,3);
I3 = zeros(3,3);
I4 = zeros(3,3);
I5 = zeros(3,3);
I6 = zeros(3,3);
I6(3,3) = 0.0002;
I_box = (box_mass/600) * sym(eye(3));

R01 = T01(1:3,1:3);
R02 = T02(1:3,1:3);
R03 = T03(1:3,1:3);
R04 = T04(1:3,1:3);
R05 = T05(1:3,1:3);
R06 = T06(1:3,1:3);

I01 = simplify(R01*I1*R01.', 'Steps', 10);
I02 = simplify(R02*I2*R02.', 'Steps', 10);
I03 = simplify(R03*I3*R03.', 'Steps', 10);
I04 = simplify(R04*I4*R04.', 'Steps', 10);
I05 = simplify(R05*I5*R05.', 'Steps', 10);
I06 = simplify(R06*I6*R06.', 'Steps', 10);
I0_box = simplify(R06 * I_box * R06.', 'Steps', 10);

%Linear Kinetic Energy
Kv1 = (sym(1)/2)*m1*(vc1.'*vc1);
Kv2 = (sym(1)/2)*m2*(vc2.'*vc2);
Kv3 = (sym(1)/2)*m3*(vc3.'*vc3);
Kv4 = (sym(1)/2)*m4*(vc4.'*vc4);
Kv5 = (sym(1)/2)*m5*(vc5.'*vc5);
Kv6 = (sym(1)/2)*m6*(vc6.'*vc6);
%Kv_box = (sym(1)/2)*box_mass*(vc_box.'*vc_box);
%Kv_box = simplify(Kv_box, 'Steps', 5);

%Rotational Kinetic Energy
Kw1 = simplify((1/2)*(omega1.'*I01*omega1), 'Steps', 10);
Kw2 = simplify((1/2)*(omega2.'*I02*omega2), 'Steps', 10);
Kw3 = simplify((1/2)*(omega3.'*I03*omega3), 'Steps', 10);
Kw4 = simplify((1/2)*(omega4.'*I04*omega4), 'Steps', 10);
Kw5 = simplify((1/2)*(omega5.'*I05*omega5), 'Steps', 10);
Kw6 = simplify((sym(1)/2)*(omega6.'*I06*omega6), 'Steps', 10);
%Kw_box = simplify((sym(1)/2)*(omega_box.'*I0_box*omega_box), 'Steps', 10);

%Total Kinetic Energy of Each Link
K1 = Kv1 + Kw1;
K2 = Kv2 + Kw2;
K3 = Kv3 + Kw3;
K4 = Kv4 + Kw4;
K5 = Kv5 + Kw5;
K6 = Kv6 + Kw6;
%K_box = Kv_box + Kw_box;

% Total Kinetic Energy
K_total = simplify(K1 + K2 + K3 + K4 + K5 + K6, 'Steps', 10);

% Potential Energy of Each Link
P1 = simplify(-m1*g0.'*pc1, 'Steps', 10);
P2 = simplify(-m2*g0.'*pc2, 'Steps', 10);
P3 = simplify(-m3*g0.'*pc3, 'Steps', 10);
P4 = simplify(-m4*g0.'*pc4, 'Steps', 10);
P5 = simplify(-m5*g0.'*pc5, 'Steps', 10);
P6 = simplify(-m6*g0.'*pc6, 'Steps', 10);
P_box = simplify(-box_mass*g0.' * pc_box, 'Steps', 10);

% Total Potential Energy
P_total = simplify(P1 + P2 + P3 + P4 + P5 + P6, ...
    'Steps', 10);

% Lagrangian
L = simplify(K_total - P_total, 'Steps', 10);
%L_box = K_box - P_box;

q_col = q.';

%Euler-lagrange Equations
dL_dqdot = jacobian(L, q_dot).';
dL_dq = jacobian(L, q_col).';
%dLbox_dqdot = jacobian(L_box, q_dot).';
%dLbox_dq = jacobian(L_box, q_col).';

d_dt_dL_dqdot = jacobian(dL_dqdot, q_col)*q_dot + ...
    jacobian(dL_dqdot, q_dot)*q_ddot;
%d_dt_dLbox_dqdot = jacobian(dLbox_dqdot, q_col) * q_dot + ...
%    jacobian(dLbox_dqdot, q_dot) * q_ddot;

tau = simplify(d_dt_dL_dqdot - dL_dq, 'Steps', 50);
%tau_box = simplify(d_dt_dLbox_dqdot - dLbox_dq, 'Steps', 10);


%Compact dynamic model
M = simplify(jacobian(tau, q_ddot), 'Steps', 50);
G = simplify(jacobian(P_total, q_col).', 'Steps', 50);

M_box = box_mass *(Jv_box.' * Jv_box) + Jw_box.'*I0_box*Jw_box;
G_box = jacobian(P_box, q_col).';

%% Coriolis matrix using Christoffel coefficients
C = sym(zeros(6,6));

for i = 1:6
    for j = 1:6

        Cij = sym(0);

        for k = 1:6

            Cijk = (sym(1)/2)*( ...
                diff(M(i,j), q_col(k)) + ...
                diff(M(i,k), q_col(j)) - ...
                diff(M(j,k), q_col(i)) );

            Cij = Cij + Cijk*q_dot(k);
        end

        C(i,j) = simplify(Cij, 'Steps', 10);
    end
end

C_box = sym(zeros(6,6));

for i = 1:6
    for j = 1:6

        Cij_box = sym(0);

        for k = 1:6
            Cijk_box = (sym(1)/2)*(diff(M_box(i,j), q_col(k)) + ...
                diff(M_box(i,k), q_col(j)) - ...
                diff(M_box(j,k), q_col(i)) );
            
            Cij_box = Cij_box + Cijk_box*q_dot(k);
        end

        C_box(i,j) = simplify(Cij_box, 'Steps', 10);
    end
end

Cqdot = simplify(C*q_dot, 'Steps', 10);

Cqdot_box = simplify(C_box*q_dot, 'Steps', 10);

M_display = simplify(vpa(M,6));
Cqdot_display = simplify(vpa(Cqdot, 6));
G_display = simplify(vpa(G,6));

%% Numerical evaluation at home

q_home = zeros(6,1);
qdot_home = zeros(6,1);

M_home = double(subs(M, q_col, q_home));

C_home = double(subs(C, ...
    [q_col; q_dot], ...
    [q_home; qdot_home]));

G_home = double(subs(G, q_col, q_home));

M_box_zero = simplify(subs(M_box, box_mass, 0));
C_box_zero = simplify(subs(C_box, box_mass, 0));
G_box_zero = simplify(subs(G_box, box_mass, 0));

disp('M_box at zero mass:')
disp(M_box_zero)

disp('C_box at zero mass:')
disp(C_box_zero)

disp('G_box at zero mass:')
disp(G_box_zero)

%% Step 4 - Nonzero numerical validation

q_test = [
     0.2;
    -0.5;
     0.7;
    -0.3;
     0.4;
    -0.2
];

qdot_test = [
     0.10;
    -0.20;
     0.15;
     0.05;
    -0.10;
     0.08
];

qddot_test = [
     0.30;
    -0.10;
     0.20;
     0.05;
    -0.15;
     0.10
];

M_test = double(subs(M, q_col, q_test));

C_test = double(subs(C, ...
    [q_col; q_dot], ...
    [q_test; qdot_test]));

G_test = double(subs(G, q_col, q_test));

%% Step 7 - Generate numerical model functions

matlabFunction( ...
    M, ...
    'File', 'UR5e_M', ...
    'Vars', {q_col}, ...
    'Optimize', true);

matlabFunction( ...
    C, ...
    'File', 'UR5e_C', ...
    'Vars', {q_col, q_dot}, ...
    'Optimize', true);

matlabFunction( ...
    G, ...
    'File', 'UR5e_G', ...
    'Vars', {q_col}, ...
    'Optimize', true);

%% Validate payload compact model

box_mass_test = 1.0;

M_box_test = double(subs( ...
    M_box, ...
    [q_col; box_mass], ...
    [q_test; box_mass_test]));

C_box_test = double(subs( ...
    C_box, ...
    [q_col; q_dot; box_mass], ...
    [q_test; qdot_test; box_mass_test]));

G_box_test = double(subs( ...
    G_box, ...
    [q_col; box_mass], ...
    [q_test; box_mass_test]));

tau_box_compact_test = ...
    M_box_test*qddot_test + ...
    C_box_test*qdot_test + ...
    G_box_test;

matlabFunction( ...
    M_box, ...
    'File', 'UR5e_payload_M', ...
    'Vars', {q_col, box_mass}, ...
    'Optimize', true);

matlabFunction( ...
    C_box, ...
    'File', 'UR5e_payload_C', ...
    'Vars', {q_col, q_dot, box_mass}, ...
    'Optimize', true);

matlabFunction( ...
    G_box, ...
    'File', 'UR5e_payload_G', ...
    'Vars', {q_col, box_mass}, ...
    'Optimize', true);
