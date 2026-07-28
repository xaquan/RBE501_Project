% Compare payload torque magnitudes on one joint trajectory.

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', 'libraries', 'DynamicalModel'));

q_start = deg2rad([0, -90, 90, -90, -90, 0]);
q_goal = deg2rad([-55.9, -55.9, 80.8, -114.9, -90, -145.9]);
duration = 3.5;
sample_period = 0.032;
payload_masses = [0.0, 0.5, 1.0];

time = (0:sample_period:duration)';
if time(end) < duration
    time = [time; duration];
end

s = time / duration;
blend = 10 * s.^3 - 15 * s.^4 + 6 * s.^5;
blend_dot = (30 * s.^2 - 60 * s.^3 + 30 * s.^4) / duration;
blend_ddot = (60 * s - 180 * s.^2 + 120 * s.^3) / duration^2;

delta_q = q_goal - q_start;
q = q_start + blend * delta_q;
q_dot = blend_dot * delta_q;
q_ddot = blend_ddot * delta_q;

torques = zeros(numel(time), 6, numel(payload_masses));

for mass_index = 1:numel(payload_masses)
    box_mass = payload_masses(mass_index);

    for sample_index = 1:numel(time)
        q_sample = q(sample_index, :)';
        q_dot_sample = q_dot(sample_index, :)';
        q_ddot_sample = q_ddot(sample_index, :)';

        M = UR5e_M(q_sample) + UR5e_payload_M(q_sample, box_mass);
        C = UR5e_C(q_sample, q_dot_sample) + ...
            UR5e_payload_C(q_sample, q_dot_sample, box_mass);
        G = UR5e_G(q_sample) + UR5e_payload_G(q_sample, box_mass);

        torques(sample_index, :, mass_index) = ...
            (M * q_ddot_sample + C * q_dot_sample + G)';
    end
end

torques = abs(torques);

figure('Name', 'Payload Torque Comparison', 'Color', 'w');
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for joint_index = 1:6
    nexttile;
    plot(time, torques(:, joint_index, 1), 'LineWidth', 1.5);
    hold on;
    plot(time, torques(:, joint_index, 2), '--', 'LineWidth', 1.5);
    plot(time, torques(:, joint_index, 3), ':', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('Joint %d torque (N m)', joint_index));
    title(sprintf('Joint %d', joint_index));
    legend('0 kg', '0.5 kg', '1.0 kg', 'Location', 'best');
end

sgtitle('UR5e Torque Comparison on the Same Trajectory');
