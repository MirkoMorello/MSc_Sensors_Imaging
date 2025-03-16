%% INITIALIZE MODTRAN PARALLEL
% Clear MATLAB workspace and set MODTRAN path
clearvars m MODTRANExe MODTRANPath;

% MODTRANPath path
MODTRANPath = 'MODTRAN5/bin/';

% MODTRANExe
if ispc == 1        % Windows
    MODTRANExe = 'MODTRAN5/bin/Mod5_win64.exe';
elseif isunix == 1  % Linux
    MODTRANExe = 'MODTRAN5/bin/Mod5_linux.exe';
else                % Mac
    MODTRANExe = 'MODTRAN5/bin/Mod5_mac.exe';
end

save('MODTRAN5/matlab-modtran-5-aba70d781805/MODTRANExe.mat', 'MODTRANPath', 'MODTRANExe');



%% Enable parallel computation
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    
    poolobj = parpool('local');    
    poolsize = poolobj.NumWorkers;
else
    poolsize = poolobj.NumWorkers;
end

[~,hostname]= system('hostname');

disp('----------------------------------------------------');
disp('Parallel processing enabled ->');
disp(['-> ' num2str(poolsize) ' workers active on ' hostname]);
disp('----------------------------------------------------');
%% MODTRAN
% -- SPECTRAL RANGE
%srange = [400 2500];               % spectral range VNIR-SWIR
srange = [650 850];				   % hyplant fluo



% -- ATMOSPHERIC PARAMs. [MODTRAN5]
PARMS.ATM.LSUNFL = 1;

% -- Atmospheric Parameters
PARMS.ATM.AOT       = 0.01;          % Aerosol Optical Depth @550nm
PARMS.ATM.MODEL     = 2;              % Mid-Latitude Winter (45  North Latitude).
PARMS.ATM.IHAZE     = 0;              % Aerosol vertical profile
PARMS.ATM.CDASTM    = ' ';            % Aerosol Angstrom Law inputs
PARMS.ATM.ASTMX     = 0;              % Angstrom parameter
PARMS.ATM.IPH       = 2;              % Aerosol phase function
PARMS.ATM.G         = 0;              % Henyey-Greenstein phase function Asymmetry Parameter [-1 1]
%PARMS.ATM.H2OSTR    = SUR.H2O(i)*1.01972;  % Vertical water vapor (SUR[hPa]*1.01972 = [gcm-2])
PARMS.ATM.H2OSTR    = 0.5;

% -- Geometric/Observational Parameters
PARMS.ATM.H1        = 100;
PARMS.ATM.SZA       = 27;
PARMS.ATM.SAA       = 170;
PARMS.ATM.RAA       = 0;
PARMS.ATM.VZA       = 8.5;
PARMS.ATM.IDAY      = 191;         % DOY
PARMS.ATM.GNDALT    = 0.01;       % GNDALT - ground altitude

PARMS.ATM.WVL       = [srange(1) srange(2)];

%% RUN MODTRAN SIMULATION
% Run MODTRAN simulation to obtain the transfer functions.
[wvlLUT, T14, DV] = run_MODTRAN5_AC_PAR_4RUNS('./', 'MODTRAN_SCOPE_ATM', PARMS, 'FLUO', 'A', 1, 0, 0);
for i = 1:14
    eval(sprintf('t%d = T14(:,%d);', i, i));
end

%% LOAD WAVELENGTHS, REFLECTANCE, AND FLUORESCENCE DATA
wlS = load('wlS.txt');  % Assumes wlS.txt contains numeric wavelengths
wlF = load('wlF.txt');  % Assumes wlS.txt contains numeric wavelengths

% Use readmatrix (available in R2019a and later) to load data while handling header lines
R_data   = readmatrix('reflectance.csv');    % Size: [n x numObs]
SIF_data = readmatrix('fluorescence.csv');     % Size: [n x numObs]

% Interpolate each observation onto the MODTRAN wavelength grid (wvlLUT)
R = interp1(wlS, R_data', wvlLUT)';  
F = interp1(wlF, SIF_data', wvlLUT)';

%% COMPUTE BOTTOM-OF-ATMOSPHERE RADIANCE (LBOA)
% Here we assume LBOA is the sum of the reflectance signal and the fluorescence.
%LBOA = t1 .* (t4 .* R + ((t5 + t2 .* R) ./ (1 - R .* t3)) + F + ((F .* t3 .* R) ./ (1 - R .* t3)));


%% COMPUTE WHITE LAMBERTIAN REFERENCE RADIANCE (LWLR)
%LWLR = t1 .* (t4 + (t5 + t12 .* R) / (1 - R .* t3)) + ((F .* t3) ./ (1 - R .* t3));

%% COMPUTE APPARENT REFLECTANCE (rho_app) AS PER EQ. (5)
%rho_app = LBOA ./ LWLR;  % Element-wise division for each observation

%% Compute the LTOA
%LTOA = (t1 .* t2) + ((t1 .* (t8 .* rho_app + t9 .* rho_app + t10 .* rho_app + t11 .* rho_app) + (t6 * F + t7 * F)) ./ (1 - t3 .* R));

%% DEFINING PARAMETERS RANGE

GNDALT_values = [0.01];  % Example values (meters)
SZA_values    = [27];        % Example values (degrees)
AOT_values    = [0.01];      % Example values


%% ITERATING OVER ALL THE POSSIBLE COMBINATIONS
[gridGNDALT, gridSZA, gridAOT] = ndgrid(GNDALT_values, SZA_values, AOT_values);

combinations = [gridGNDALT(:), gridSZA(:), gridAOT(:)];

for idx = 1:size(combinations, 1)
    currentGNDALT = combinations(idx, 1);
    currentSZA    = combinations(idx, 2);
    currentAOT    = combinations(idx, 3);
    
    % Process your combination here:
    fprintf('Processing GNDALT = %g, SZA = %g, AOT = %g\n', currentGNDALT, currentSZA, currentAOT);
    
    [wvlLUT_new, T14, DV] = run_MODTRAN5_AC_PAR_4RUNS('./', 'MODTRAN_SCOPE_ATM', PARMS, 'FLUO', 'A', 1, 0, 0);
    for m = 1:14
        eval(sprintf('t%d = T14(:,%d);', m, m));
    end

    % Here we assume LBOA is the sum of the reflectance signal and the fluorescence.
    % Equation 3
    LBOA = t1' .* (t4' .* R + ((t5' + t2' .* R) ./ (1 - R .* t3')) + F + ((F .* t3' .* R) ./ (1 - R .* t3')));
    
    % Equation 4
    LWLR = t1' .* (t4' + (t5' + t12' .* R) ./ (1 - R .* t3')) + ((F .* t3') ./ (1 - R .* t3'));

    % Equation 5
    rho_app = LBOA ./ LWLR;  % Element-wise division for each observation

    % Equation 7
    LTOA = (t1' .* t2') + ((t1' .* (t8' .* rho_app + t9' .* rho_app + t10' .* rho_app + t11' .* rho_app) + (t6' .* F + t7' .* F)) ./ (1 - t3' .* R));

    filename_LTOA = sprintf('LTOA_GNDALT_%g_SZA_%g_AOT_%g.csv', PARMS.ATM.GNDALT, PARMS.ATM.SZA, PARMS.ATM.AOT);

    csvwrite(filename_LTOA, [wvlLUT_new, LTOA']);

end
