function [reflectance, fluorescence] = run_scope_wrapper(input_struct)
    % Input validation
    if ~isfield(input_struct.leafbio, 'Cab') || ~isfield(input_struct.canopy, 'LAI')
        error('Missing required parameters');
    end
    
    % Create parameter structure with defaults
    params = struct();
    
    % Leaf parameters
    params.leafbio = struct(...
        'Cab', input_struct.leafbio.Cab,...
        'Cca', 40,...  % Default carotenoid content
        'Cw', 0.01...  % Default leaf water content
    );
    
    % Canopy parameters
    params.canopy = struct(...
        'LAI', input_struct.canopy.LAI,...
        'LIDFa', -0.35...  % Default leaf angle distribution
    );
    
    % Observation geometry
    params.angles = struct(...
        'tts', input_struct.angles.tts...  % Solar zenith angle
    );
    
    % Soil properties
    params.soil = struct(...
        'spectrum', rand(1, 2101)...  % Random soil spectrum
    );
    
    % Simulation settings
    params.simulation = struct(...
        'radiation', 'solar',...
        'calc_fluor', true...
    );
    
    % Spectral settings (HyPlant FLUO range)
    params.spectral = struct(...
        'wlF', (670:0.1:780)',...  % Fluorescence wavelengths
        'wlS', (670:0.1:780)'...   % Reflectance wavelengths
    );
    
    % Run SCOPE
    [~, RTM] = SCOPE(params);
    
    % Return outputs
    reflectance = RTM.spectrum(:,1);
    fluorescence = RTM.spectrum(:,2);
end
