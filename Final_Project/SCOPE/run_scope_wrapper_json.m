function run_scope_wrapper_json()
    try
        % Minimal path setup
        scope_dir = fileparts(mfilename('fullpath'));
        addpath(genpath(fullfile(scope_dir, 'src')));
        
        % Run SCOPE - assumes input_data.csv already exists
        SCOPE; 
        
    catch ME
        error('SCOPE_ERROR: %s', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end
end