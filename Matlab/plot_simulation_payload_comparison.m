% Compare the full Webots pickup, carry, and placement runs.

script_dir = fileparts(mfilename('fullpath'));
log_path = fullfile(script_dir, '..', 'torque_logs', ...
    'simulation_command_torque.csv');

if ~isfile(log_path)
    error('Run the Webots controller first: %s', log_path);
end

data = readtable(log_path);
required = {'time', 'run', 'phase', 'payload_mass', ...
    'tau1_cmd', 'tau2_cmd', 'tau3_cmd', ...
    'tau4_cmd', 'tau5_cmd', 'tau6_cmd'};
if ~all(ismember(required, data.Properties.VariableNames))
    error('Unexpected simulation torque-log format.');
end

[time_zero, torque_zero] = benchmark_run(data, 0.0, 0);
[time_half, torque_half] = benchmark_run(data, 0.5, 1);
[time_full, torque_full] = benchmark_run(data, 1.0, 2);

figure('Name', 'Webots Payload Torque Comparison', 'Color', 'w');
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for joint_index = 1:6
    nexttile;
    plot(time_zero, torque_zero(:, joint_index), ':', 'LineWidth', 1.5);
    hold on;
    plot(time_half, torque_half(:, joint_index), 'LineWidth', 1.5);
    plot(time_full, torque_full(:, joint_index), '--', 'LineWidth', 1.5);
    grid on;
    xlabel('Full pick-place run time (s)');
    ylabel(sprintf('Joint %d torque (N m)', joint_index));
    title(sprintf('Joint %d', joint_index));
    legend('0 kg', '0.5 kg', '1.0 kg', 'Location', 'best');
end

sgtitle('Commanded Webots Torque: Same Pickup, Carry, and Place Path');


function [relative_time, torque] = benchmark_run(data, payload_mass, run_number)
    indices = find(data.run == run_number & ...
        abs(data.payload_mass - payload_mass) < 1e-6);

    if isempty(indices)
        error('No %.1f kg benchmark run found in the log.', payload_mass);
    end

    relative_time = data.time(indices) - data.time(indices(1));
    torque = abs([data.tau1_cmd(indices), data.tau2_cmd(indices), ...
        data.tau3_cmd(indices), data.tau4_cmd(indices), ...
        data.tau5_cmd(indices), data.tau6_cmd(indices)]);
end
