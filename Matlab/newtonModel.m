clear;
clc;

%% ============================================================
% 1. Symbolic variables
% =============================================================
syms q1 q2 q3 real
syms qdot1 qdot2 qdot3 real
syms qddot1 qddot2 qddot3 real

syms m1 m2 m3 real
syms g real

syms l1 l2 l3 real
syms lc1 lc2 lc3 real

q     = [q1; q2; q3];
qdot  = [qdot1; qdot2; qdot3];
qddot = [qddot1; qddot2; qddot3];

l  = [l1; l2; l3];
lc = [lc1; lc2; lc3];

joint_count = length(q);

%% ============================================================
% 2. Link inertia matrices
% =============================================================
syms Ixx1 Iyy1 Izz1 Ixy1 Ixz1 Iyz1 real
I1 = [
    Ixx1, Ixy1, Ixz1;
    Ixy1, Iyy1, Iyz1;
    Ixz1, Iyz1, Izz1
];

syms Ixx2 Iyy2 Izz2 Ixy2 Ixz2 Iyz2 real
I2 = [
    Ixx2, Ixy2, Ixz2;
    Ixy2, Iyy2, Iyz2;
    Ixz2, Iyz2, Izz2
];

syms Ixx3 Iyy3 Izz3 Ixy3 Ixz3 Iyz3 real
I3 = [
    Ixx3, Ixy3, Ixz3;
    Ixy3, Iyy3, Iyz3;
    Ixz3, Iyz3, Izz3
];

I = {I1, I2, I3};

%% ============================================================
% 3. Rotation matrices
% =============================================================

Rotz = @(theta) [
    cos(theta), -sin(theta), 0;
    sin(theta),  cos(theta), 0;
    0,           0,          1
];

Roty = @(theta) [
     cos(theta), 0, sin(theta);
     0,          1, 0;
    -sin(theta), 0, cos(theta)
];

% Relative rotations:
% R01 maps vectors from frame 1 to frame 0
% R12 maps vectors from frame 2 to frame 1
% R23 maps vectors from frame 3 to frame 2
R01 = Rotz(q1);
R12 = Roty(-q2);
R23 = Roty(-q3);

% World-to-link cumulative orientations
R02 = simplify(R01 * R12);
R03 = simplify(R02 * R23);

R_relative = {R01, R12, R23};
R0_i       = {R01, R02, R03};

%% ============================================================
% 4. Joint axes
% =============================================================

% Joint axes expressed in each joint's local frame
joint_axis = {
    [0;  0; 1], ...   % Joint 1: +z
    [0; -1; 0], ...   % Joint 2: -y
    [0; -1; 0]  ...   % Joint 3: -y
};

% Optional: joint axes expressed in world frame
axis_world = cell(1, joint_count);

axis_world{1} = joint_axis{1};
axis_world{2} = simplify(R01 * joint_axis{2});
axis_world{3} = simplify(R02 * joint_axis{3});

%% ============================================================
% 5. Angular velocity and angular acceleration
% =============================================================

w     = repmat({zeros(3,1)}, 1, joint_count);
alpha = repmat({zeros(3,1)}, 1, joint_count);

w_pre     = zeros(3,1);
alpha_pre = zeros(3,1);

for i = 1:joint_count

    % Transform previous angular quantities into current frame
    Ri_T = R_relative{i}.';

    w_pre_in_i     = Ri_T * w_pre;
    alpha_pre_in_i = Ri_T * alpha_pre;

    axis_i = joint_axis{i};

    % Angular velocity
    w{i} = simplify( ...
        w_pre_in_i + axis_i*qdot(i));

    % Angular acceleration
    alpha{i} = simplify( ...
        alpha_pre_in_i ...
        + axis_i*qddot(i) ...
        + cross(w_pre_in_i, axis_i*qdot(i)));

    w_pre     = w{i};
    alpha_pre = alpha{i};
end

%% ============================================================
% 6. COM and endpoint position vectors
% =============================================================

% Vectors from joint origin O_i to COM C_i,
% expressed in frame i
r_ic = {
    [0;   0; lc1], ...  % O1 -> C1
    [lc2; 0; 0],   ...  % O2 -> C2
    [lc3; 0; 0]    ...  % O3 -> C3
};

% Vectors from joint origin O_i to endpoint O_{i+1},
% expressed in frame i
r_ie = {
    [0;  0; l1], ...    % O1 -> O2
    [l2; 0; 0],  ...    % O2 -> O3
    [l3; 0; 0]   ...    % O3 -> end effector
};

%% ============================================================
% 7. Linear acceleration of COM and endpoints
% =============================================================

ac = repmat({zeros(3,1)}, 1, joint_count);
ae = repmat({zeros(3,1)}, 1, joint_count);
gi = repmat({zeros(3,1)}, 1, joint_count);

% Base-origin acceleration, excluding gravity
ae_pre = zeros(3,1);

% Gravity expressed in world frame
g0 = [0; 0; -g];

for i = 1:joint_count

    Ri_T = R_relative{i}.';

    % Acceleration of origin O_i expressed in frame i
    a_origin_i = Ri_T * ae_pre;

    % COM acceleration
    ac{i} = simplify( ...
        a_origin_i ...
        + cross(alpha{i}, r_ic{i}) ...
        + cross(w{i}, cross(w{i}, r_ic{i})));

    % Link endpoint acceleration
    ae{i} = simplify( ...
        a_origin_i ...
        + cross(alpha{i}, r_ie{i}) ...
        + cross(w{i}, cross(w{i}, r_ie{i})));

    % Gravity expressed in frame i
    gi{i} = simplify(R0_i{i}.' * g0);

    % Endpoint becomes the next joint origin
    ae_pre = ae{i};
end

%% ============================================================
% 8. Newton-Euler backward recursion
% =============================================================

mass = [m1; m2; m3];

% Force and moment transmitted through each link
f = repmat({zeros(3,1)}, 1, joint_count);
n_link = repmat({zeros(3,1)}, 1, joint_count);

% Joint torque vector
tau = sym(zeros(joint_count,1));

% Force and moment from a nonexistent link 4
f_child = zeros(3,1);
n_child = zeros(3,1);

for i = joint_count:-1:1

    % ---------------------------------------------------------
    % Inertial force acting at the center of mass
    %
    % ac{i} does not include gravity, so subtract gi{i}
    % ---------------------------------------------------------
    F_i = simplify( ...
        mass(i) * (ac{i} - gi{i}) ...
    );

    % ---------------------------------------------------------
    % Rotational inertial moment about the COM
    % ---------------------------------------------------------
    N_i = simplify( ...
        I{i} * alpha{i} ...
        + cross(w{i}, I{i} * w{i}) ...
    );

    if i == joint_count
        % Last link has no child
        f_child_in_i = zeros(3,1);
        n_child_in_i = zeros(3,1);
    else
        % R_relative{i+1} maps a vector from frame i+1 to frame i
        R_i_ip1 = R_relative{i+1};

        f_child_in_i = simplify(R_i_ip1 * f_child);
        n_child_in_i = simplify(R_i_ip1 * n_child);
    end

    % ---------------------------------------------------------
    % Total force transmitted through link i
    % ---------------------------------------------------------
    f{i} = simplify( ...
        F_i + f_child_in_i ...
    );

    % ---------------------------------------------------------
    % Total moment about joint origin O_i
    % ---------------------------------------------------------
    n_link{i} = simplify( ...
        N_i ...
        + n_child_in_i ...
        + cross(r_ic{i}, F_i) ...
        + cross(r_ie{i}, f_child_in_i) ...
    );

    % ---------------------------------------------------------
    % Project link moment onto joint axis
    % ---------------------------------------------------------
    tau(i) = simplify( ...
        joint_axis{i}.' * n_link{i} ...
    );

    % Pass values to parent link
    f_child = f{i};
    n_child = n_link{i};
end

tau = simplify(tau);

disp('Joint torque vector tau:');
pretty(tau);

%% ============================================================
% 9. Extract inertia matrix D(q)
% =============================================================

D = simplify(jacobian(tau, qddot));

% Apply additional symbolic simplification
D = simplify((D));

disp('Inertia matrix D(q):');
pretty(D);

D_symmetry_error = simplify(D - D.');

disp('D(q) - D(q)^T:');
pretty(D_symmetry_error);

%% ============================================================
% 10. Extract gravity vector G(q)
% =============================================================

zero_qdot  = zeros(joint_count,1);
zero_qddot = zeros(joint_count,1);

G = simplify(subs( ...
    tau, ...
    [qdot; qddot], ...
    [zero_qdot; zero_qddot] ...
));

disp('Gravity vector G(q):');
pretty(G);

%% ============================================================
% 11. Calculate Coriolis matrix C(q, qdot)
% =============================================================

C = sym(zeros(joint_count, joint_count));

for i = 1:joint_count
    for j = 1:joint_count

        C_ij = sym(0);

        for k = 1:joint_count

            christoffel_ijk = sym(1)/2 * ( ...
                diff(D(i,j), q(k)) ...
                + diff(D(i,k), q(j)) ...
                - diff(D(j,k), q(i)) ...
            );

            C_ij = C_ij + christoffel_ijk * qdot(k);
        end

        C(i,j) = simplify(C_ij);
    end
end

C = simplify((C));

disp('Coriolis and centrifugal matrix C(q,qdot):');
pretty(C);

%% ============================================================
% 12. Coriolis and centrifugal torque vector
% =============================================================

Cqdot = simplify(C * qdot);

disp('Coriolis/centrifugal vector C(q,qdot)*qdot:');
pretty(Cqdot);

%% ============================================================
% 13. Verify tau = D*qddot + C*qdot + G
% =============================================================

tau_DCG = simplify( ...
    D * qddot ...
    + C * qdot ...
    + G ...
);

verification_error = simplify( ...
    (tau - tau_DCG) ...
);

disp('Reconstructed torque D*qddot + C*qdot + G:');
pretty(tau_DCG);

disp('Verification error: tau - (D*qddot + C*qdot + G)');
pretty(verification_error);