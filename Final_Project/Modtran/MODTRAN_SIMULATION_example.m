
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
srange = [650 800];				   % hyplant fluo



% -- ATMOSPHERIC PARAMs. [MODTRAN5]
PARMS.ATM.LSUNFL = 1;

% -- Atmospheric Parameters
PARMS.ATM.AOT       = 0.01;          % Aerosol Optical Depth @550nm
PARMS.ATM.MODEL     = 2;              % Mid-Latitude Winter (45° North Latitude).
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




%% ATMOSPHERIC SIMULATIONS (MODTRAN5)


[wvlLUT,T14,DV] = run_MODTRAN5_AC_PAR_4RUNS('./', 'MODTRAN_SCOPE_ATM',...
    PARMS,'FLUO','A', 1, 0, 0);                                      % DEBUG

% EXTRACT T14
for i=1:14; eval(['t' num2str(i) '= T14(:,' num2str(i) ');']); end


%% SURFACE REFLECTANCE (load file)

fname = "E:\google_drive_unimib\SNOW\DATA\cal_val_new.xlsx";

T = readtable(fname,'Sheet','plateau_rosa');



%% SURFACE/ATMOSPHERE RT COUPLING

wvl = R(:,1);
%R = R(:,2:end);
R = R(:,[2 5]);
R = interp1(wvl, R, wvlLUT);

LSUN    = t1.*t4;
LSKY    = t1.*t5;
LBOA    = t1.*(t4+t5)  .* R;
%LTOA    = t1.*t2 + (t1.*(t4+t5)  .* R .* (t6+t7)) ./ (1-R.*t3);
% Bayat
LTOA = (t1.*t2) + t1.*t8.*R + t1.* ( ( (t9).*R + t10.* R + t11.*R) ./ (1-R.*t3));








%% SPECTRAL CONVOLUTION

% LSUN_   = convolve_ISRF(wvlLUT,LSUN,  prs_wvl, prs_fwhm);
% LSKY_   = convolve_ISRF(wvlLUT,LSKY,  prs_wvl, prs_fwhm);
% LBOA_   = convolve_ISRF(wvlLUT,LBOA,  prs_wvl, prs_fwhm);
% LTOA_   = convolve_ISRF(wvlLUT,LTOA,  prs_wvl, prs_fwhm);
% 
% LSKYa_  = convolve_ISRF(wvlLUT,LSKYa, prs_wvl, prs_fwhm);
% LBOAa_  = convolve_ISRF(wvlLUT,LBOAa, prs_wvl, prs_fwhm);
% LTOAa_  = convolve_ISRF(wvlLUT,LTOAa, prs_wvl, prs_fwhm);


