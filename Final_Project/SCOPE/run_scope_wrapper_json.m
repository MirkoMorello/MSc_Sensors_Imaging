function json_output = run_scope_wrapper_json(input_path, output_path)
    % Get absolute paths (MATLAB's working directory is SCOPE)
    scope_dir = fileparts(mfilename('fullpath'));
    output_dir = fullfile(scope_dir, 'output');


    try:
        % Add required paths
        addpath(genpath(fullfile(scope_dir, 'src')));

        % Run SCOPE in base workspace
        evalin('base', 'SCOPE');

        % Collect results from output directory (Simplified)
        result = struct();
        try:
            spectrum = csvread(fullfile(output_dir, 'reflectance.csv'));
            result.spectrum = struct(...
                'wavelength', spectrum(:,1),...
                'reflectance', spectrum(:,2),...
                'fluorescence', csvread(fullfile(output_dir, 'fluorescence.csv'))...
            );
        catch ME
            error('Failed to read output files: %s', ME.message);
        end


        % Write JSON
        json_output = jsonencode(result);
        fid = fopen(output_path, 'w');
        if fid == -1, error('Failed to create output file'); end
        fprintf(fid, '%s', json_output);
        fclose(fid);

    catch ME
        rethrow(ME);
    end
end