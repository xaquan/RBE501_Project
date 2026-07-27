function [D, C, G, tau, details] = ur5e_newton_euler_dcg(q, qdot, qddot)
%UR5E_NEWTON_EULER_DCG Numerical D(q), C(q,qdot), G(q) for Webots UR5e.
%
%   [D,C,G,tau,details] = ur5e_newton_euler_dcg(q,qdot,qddot)
%
% Robot equation:
%       tau = D(q)*qddot + C(q,qdot)*qdot + G(q)
%
% This implementation:
%   1. Uses a world-frame recursive Newton-Euler inverse-dynamics method.
%   2. Obtains D by six inverse-dynamics calls with unit joint acceleration.
%   3. Obtains G from one static inverse-dynamics call with gravity.
%   4. Forms C from numerical Christoffel symbols of D.
%   5. Avoids Symbolic Math Toolbox and Robotics System Toolbox.
%
% IMPORTANT:
% The supplied Webots PROTO gives masses and centers of mass, but it does
% not explicitly provide inertia tensors. Webots can infer inertia from
% collision geometry. The approximate diagonal inertia tensors below use
% simple cylinder models. Replace model.Ibody(:,:,i) with measured or URDF
% inertia tensors when exact values are available.
%
% Joint order:
%   q1 shoulder_pan_joint
%   q2 shoulder_lift_joint
%   q3 elbow_joint
%   q4 wrist_1_joint
%   q5 wrist_2_joint
%   q6 wrist_3_joint
%
% Example:
%   q    = deg2rad([0; -90; 90; -90; -90; 0]);
%   qdot = zeros(6,1);
%   qddot = zeros(6,1);
%   [D,C,G,tau,details] = ur5e_newton_euler_dcg(q,qdot,qddot);
%   disp(D); disp(C); disp(G); disp(tau);
%
% To use a point-mass model only:
%   Set model.Ibody(:) = 0 inside createUR5eModel().

    if nargin == 0
        q     = deg2rad([0; -90; 90; -90; -90; 0])
        qdot  = zeros(6,1);
        qddot = zeros(6,1);
    elseif nargin ~= 3
        error('Use ur5e_newton_euler_dcg(q, qdot, qddot), with 6x1 vectors.');
    end

    q     = validateJointVector(q,     'q');
    qdot  = validateJointVector(qdot,  'qdot');
    qddot = validateJointVector(qddot, 'qddot');

    model = createUR5eModel();
    n = model.n;

    zeroN = zeros(n,1);
    zeroG = zeros(3,1);

    %% D(q): unit-acceleration inverse-dynamics evaluations
    D = zeros(n,n);
    for j = 1:n
        unitAcceleration = zeros(n,1);
        unitAcceleration(j) = 1;
        D(:,j) = inverseDynamicsNE( ...
            model, q, zeroN, unitAcceleration, zeroG);
    end

    % Remove very small numerical asymmetry.
    D = 0.5 * (D + D.');

    %% G(q): static gravity torque
    G = inverseDynamicsNE( ...
        model, q, zeroN, zeroN, model.gravity);

    %% C(q,qdot): Christoffel matrix computed from numerical dD/dq
    C = coriolisMatrixFromMassMatrix(model, q, qdot);

    %% Final torque
    tau = D*qddot + C*qdot + G;

    %% Independent Newton-Euler checks
    tauNE = inverseDynamicsNE( ...
        model, q, qdot, qddot, model.gravity);

    hNE = inverseDynamicsNE( ...
        model, q, qdot, zeroN, zeroG);

    details = struct();
    details.model = model;
    details.tauNewtonEuler = tauNE;
    details.coriolisVectorNewtonEuler = hNE;
    details.coriolisVectorFromC = C*qdot;
    details.torqueError = tau - tauNE;
    details.coriolisError = C*qdot - hNE;
    details.massMatrixSymmetryError = norm(D-D.', 'fro');
    details.minimumMassMatrixEigenvalue = min(eig(0.5*(D+D.')));

    if nargout == 0
        fprintf('\nUR5e numerical Newton-Euler dynamics\n');
        fprintf('Joint order: [shoulder_pan shoulder_lift elbow wrist_1 wrist_2 wrist_3]^T\n\n');

        disp('D(q) =');
        disp(D);

        disp('C(q,qdot) =');
        disp(C);

        disp('G(q) =');
        disp(G);

        disp('tau = D*qddot + C*qdot + G =');
        disp(tau);

        disp('Direct Newton-Euler tau =');
        disp(tauNE);

        fprintf('||tau - tau_NE|| = %.6e\n', norm(tau-tauNE));
        fprintf('||C*qdot - h_NE|| = %.6e\n', norm(C*qdot-hNE));
        fprintf('Symmetry error of D = %.6e\n', details.massMatrixSymmetryError);
        fprintf('Minimum eigenvalue of D = %.6e\n', ...
            details.minimumMassMatrixEigenvalue);
    end
end


function tau = inverseDynamicsNE(model, q, qdot, qddot, gravity)
%INVERSEDYNAMICSNE World-frame recursive Newton-Euler inverse dynamics.
%
% gravity is a 3x1 acceleration vector in the fixed world frame.
% For normal Earth gravity:
%       gravity = [0;0;-9.81]
% For gravity disabled:
%       gravity = [0;0;0]

    n = model.n;

    % World-frame quantities for every moving link.
    R = zeros(3,3,n);
    p = zeros(3,n);
    omega = zeros(3,n);
    alpha = zeros(3,n);
    aOrigin = zeros(3,n);
    aCOM = zeros(3,n);
    pCOM = zeros(3,n);
    axisWorld = zeros(3,n);

    forceCOM = zeros(3,n);
    momentCOM = zeros(3,n);

    % Base state.
    Rparent = eye(3);
    pParent = zeros(3,1);
    omegaParent = zeros(3,1);
    alphaParent = zeros(3,1);
    aParent = zeros(3,1);

    %% Forward recursion
    for i = 1:n
        axisParent = model.axisParent(:,i);
        anchorParent = model.anchorParent(:,i);
        Rfixed = model.Rfixed(:,:,i);

        % Joint axis expressed in the fixed world frame.
        axisWorld(:,i) = Rparent * axisParent;

        % Child-link origin is at the hinge anchor.
        rParentToChild = Rparent * anchorParent;
        p(:,i) = pParent + rParentToChild;

        % Child-link orientation.
        Rjoint = rotationFromAxisAngle(axisParent, q(i));
        R(:,:,i) = Rparent * Rjoint * Rfixed;

        % Angular velocity and angular acceleration.
        omega(:,i) = omegaParent + axisWorld(:,i)*qdot(i);
        alpha(:,i) = alphaParent ...
            + axisWorld(:,i)*qddot(i) ...
            + cross(omegaParent, axisWorld(:,i)*qdot(i));

        % Linear acceleration of child-link origin.
        aOrigin(:,i) = aParent ...
            + cross(alphaParent, rParentToChild) ...
            + cross(omegaParent, cross(omegaParent, rParentToChild));

        % Center of mass.
        rOriginToCOM = R(:,:,i) * model.com(:,i);
        pCOM(:,i) = p(:,i) + rOriginToCOM;

        aCOM(:,i) = aOrigin(:,i) ...
            + cross(alpha(:,i), rOriginToCOM) ...
            + cross(omega(:,i), cross(omega(:,i), rOriginToCOM));

        % Newton and Euler equations at the center of mass.
        Iworld = R(:,:,i) * model.Ibody(:,:,i) * R(:,:,i).';

        forceCOM(:,i) = model.mass(i) * (aCOM(:,i) - gravity);
        momentCOM(:,i) = Iworld*alpha(:,i) ...
            + cross(omega(:,i), Iworld*omega(:,i));

        % Pass state to next link.
        Rparent = R(:,:,i);
        pParent = p(:,i);
        omegaParent = omega(:,i);
        alphaParent = alpha(:,i);
        aParent = aOrigin(:,i);
    end

    %% Backward recursion
    tau = zeros(n,1);
    transmittedForce = zeros(3,1);
    transmittedMoment = zeros(3,1);

    for i = n:-1:1
        % Add this link's inertial wrench.
        totalForce = forceCOM(:,i) + transmittedForce;

        totalMoment = momentCOM(:,i) ...
            + cross(pCOM(:,i)-p(:,i), forceCOM(:,i)) ...
            + transmittedMoment;

        % Generalized joint torque.
        tau(i) = axisWorld(:,i).' * totalMoment;

        % Shift complete subtree wrench to the parent joint origin.
        transmittedForce = totalForce;

        if i > 1
            parentOrigin = p(:,i-1);
        else
            parentOrigin = zeros(3,1);
        end

        transmittedMoment = totalMoment ...
            + cross(p(:,i)-parentOrigin, totalForce);
    end
end


function C = coriolisMatrixFromMassMatrix(model, q, qdot)
%CORIOLISMATRIXFROMMASSMATRIX Compute C using Christoffel symbols.
%
% C_ij = sum_k 0.5*(dD_ij/dq_k + dD_ik/dq_j - dD_jk/dq_i)*qdot_k
%
% A centered finite difference is used because the model is numerical.

    n = model.n;
    dD = zeros(n,n,n);

    % Angle perturbation in radians.
    step = 1e-6;

    for k = 1:n
        qPlus = q;
        qMinus = q;
        qPlus(k) = qPlus(k) + step;
        qMinus(k) = qMinus(k) - step;

        Dplus = massMatrixFromNE(model, qPlus);
        Dminus = massMatrixFromNE(model, qMinus);

        dD(:,:,k) = (Dplus-Dminus)/(2*step);
    end

    C = zeros(n,n);

    for i = 1:n
        for j = 1:n
            value = 0;
            for k = 1:n
                christoffel = 0.5 * ( ...
                    dD(i,j,k) ...
                    + dD(i,k,j) ...
                    - dD(j,k,i));
                value = value + christoffel*qdot(k);
            end
            C(i,j) = value;
        end
    end
end


function D = massMatrixFromNE(model, q)
%MASSMATRIXFROMNE Obtain D(q) from unit-acceleration NE calls.

    n = model.n;
    zeroN = zeros(n,1);
    zeroG = zeros(3,1);

    D = zeros(n,n);
    for j = 1:n
        e = zeros(n,1);
        e(j) = 1;
        D(:,j) = inverseDynamicsNE(model, q, zeroN, e, zeroG);
    end

    D = 0.5*(D+D.');
end


function model = createUR5eModel()
%CREATEUR5EMODEL Webots UR5e kinematic and approximate inertial data.
%
% Relative child pose follows the Webots HingeJoint convention:
%   T_parent_child(q) =
%       Trans(anchorParent) * Rot(axisParent,q) * RotFixed
%
% Values are taken from the supplied Webots UR5e PROTO.

    model.n = 6;

    %% Joint axes in each parent link frame
    model.axisParent = [ ...
        0  0  0  0  0  0;
        0  1  1  1  0  1;
        1  0  0  0  1  0];

    %% Hinge anchors in each parent link frame [m]
    model.anchorParent = [ ...
        0      0       0       0      0      0;
        0      0.138  -0.131   0      0.127  0;
        0.163  0       0.425   0.392  0      0.100];

    %% Fixed endpoint rotations at q = 0
    model.Rfixed = zeros(3,3,6);
    model.Rfixed(:,:,1) = eye(3);
    model.Rfixed(:,:,2) = rotY(pi/2);
    model.Rfixed(:,:,3) = eye(3);
    model.Rfixed(:,:,4) = rotY(pi/2);
    model.Rfixed(:,:,5) = eye(3);
    model.Rfixed(:,:,6) = eye(3);

    %% Moving-link masses [kg]
    model.mass = [3.761; 8.058; 2.846; 1.370; 1.300; 0.365];

    %% Centers of mass in each child-link frame [m]
    model.com = [ ...
         0        0       0      0      0      0;
        -0.02561  0       0      0.127  0      0.071;
         0.00193  0.2125  0.150  0      0.100  0];

    %% Gravity in the Webots world frame [m/s^2]
    model.gravity = [0;0;-9.81];

    %% Approximate body-frame inertia tensors [kg*m^2]
    %
    % Replace these matrices with exact tensors when available.
    % The approximations use solid cylinders aligned with each link's
    % dominant geometric direction.
    model.Ibody = zeros(3,3,6);

    model.Ibody(:,:,1) = cylinderInertia( ...
        model.mass(1), 0.059, 0.135, 'z');

    model.Ibody(:,:,2) = cylinderInertia( ...
        model.mass(2), 0.059, 0.407, 'z');

    model.Ibody(:,:,3) = cylinderInertia( ...
        model.mass(3), 0.038, 0.360, 'z');

    model.Ibody(:,:,4) = cylinderInertia( ...
        model.mass(4), 0.038, 0.127, 'y');

    model.Ibody(:,:,5) = cylinderInertia( ...
        model.mass(5), 0.038, 0.100, 'z');

    model.Ibody(:,:,6) = cylinderInertia( ...
        model.mass(6), 0.032, 0.071, 'y');
end


function I = cylinderInertia(mass, radius, lengthValue, axisName)
%CYLINDERINERTIA Inertia of a solid cylinder about its center of mass.

    Iaxis = 0.5*mass*radius^2;
    Itransverse = mass*(3*radius^2 + lengthValue^2)/12;

    switch lower(axisName)
        case 'x'
            I = diag([Iaxis, Itransverse, Itransverse]);
        case 'y'
            I = diag([Itransverse, Iaxis, Itransverse]);
        case 'z'
            I = diag([Itransverse, Itransverse, Iaxis]);
        otherwise
            error('Cylinder axis must be x, y, or z.');
    end
end


function R = rotationFromAxisAngle(axisVector, angleValue)
%ROTATIONFROMAXISANGLE Rodrigues rotation formula.

    axisVector = axisVector(:);
    axisNorm = norm(axisVector);

    if axisNorm < eps
        error('Rotation axis cannot be zero.');
    end

    axisVector = axisVector/axisNorm;
    K = skew3(axisVector);

    R = eye(3) ...
        + sin(angleValue)*K ...
        + (1-cos(angleValue))*(K*K);
end


function S = skew3(v)
%SKEW3 Return the 3x3 cross-product matrix of a 3x1 vector.

    v = v(:);
    S = [ ...
         0    -v(3)  v(2);
         v(3)  0    -v(1);
        -v(2)  v(1)  0];
end


function R = rotY(angleValue)
%ROTY Rotation matrix about y.

    c = cos(angleValue);
    s = sin(angleValue);

    R = [ ...
         c  0  s;
         0  1  0;
        -s  0  c];
end


function value = validateJointVector(value, variableName)
%VALIDATEJOINTVECTOR Ensure a finite real 6x1 joint vector.

    if ~isnumeric(value) || numel(value) ~= 6
        error('%s must contain six numeric values.', variableName);
    end

    value = value(:);

    if ~isreal(value) || any(~isfinite(value))
        error('%s must contain finite real values.', variableName);
    end
end
