from collections import defaultdict
import csv
import itertools
import os
import shutil
import json
import matlab.engine
from pathlib import Path
import numpy as np
import pandas as pd
import struct
import re 


class SCOPEWrapperMultiRun:
    def __init__(self, scalar_files, spectral_files):
        self.base_dir = Path.cwd()
        self.scope_dir = self.base_dir / "SCOPE"
        self.original_input = self.scope_dir / "input" / "input_data.csv"
        self.default_params = self._get_default_parameters()
        self.eng = None
        self.setoptions = self._get_default_setoptions() 
        self.active_setoptions = self._get_default_setoptions()
        self.scalar_files = scalar_files
        self.spectral_files = spectral_files
        


    def _get_default_setoptions(self):
        """Loads default setoptions from setoptions.csv"""
        setoptions_path = self.scope_dir / "input" / "setoptions.csv"
        setoptions = {}
        with open(setoptions_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) >= 2: 
                    key = row[0].strip()
                    value = row[1].strip()
                    try:
                        
                        value = int(value)
                    except ValueError:
                        pass 
                    setoptions[key] = value
        return setoptions

    def _get_default_parameters(self):
        """Comprehensive default parameters for SCOPE."""
        return {
            # PROSPECT
            "Cab": 40.0,
            "Cca": 10.0,
            "Cdm": 0.012,
            "Cw": 0.009,
            "Cs": 0.0,
            "Cant": 1.0,
            "Cp": 0.0,
            "Cbc": 0.0,
            "N": 1.5,
            "rho_thermal": 0.01,
            "tau_thermal": 0.01,
            # Leaf Biochemical
            "Vcmax25": 60.0,
            "BallBerrySlope": 8.0,
            "BallBerry0": 0.01,
            "Type": 0.0,
            "kV": 0.64,
            "Rdparam": 0.015,
            "Kn0": 2.48,
            "Knalpha": 2.83,
            "Knbeta": 0.114,
            # Magnani
            "Tyear": 15.0,
            "beta": 0.51,
            "kNPQs": 0.0,
            "qLs": 1.0,
            "stressfactor": 1.0,
            # Fluorescence
            "fqe": 0.01,
            # Soil
            "spectrum": 1.0,
            "rss": 500.0,
            "rs_thermal": 0.06,
            "cs": 1180.0,
            "rhos": 1800.0,
            "lambdas": 1.55,
            "SMC": 25.0,
            "BSMBrightness": 0.5,
            "BSMlat": 25.0,
            "BSMlon": 45.0,
            # Canopy
            "LAI": 3.0,
            "hc": 2.0,
            "LIDFa": -0.35,
            "LIDFb": -0.15,
            "leafwidth": 0.1,
            "Cv": 1.0,
            "crowndiameter": 1.0,
            # Meteo
            "z": 5.0,
            "Rin": 800.0,
            "Ta": 20.0,
            "Rli": 300.0,
            "p": 970.0,
            "ea": 15.0,
            "u": 2.0,
            "Ca": 410.0,
            "Oa": 209.0,
            # Aerodynamic
            "zo": 0.25,
            "d": 1.34,
            "Cd": 0.3,
            "rb": 10.0,
            "CR": 0.35,
            "CD1": 20.6,
            "Psicor": 0.2,
            "CSSOIL": 0.01,
            "rbs": 10.0,
            "rwc": 0.0,
            # Timeseries
            "startDOY": 20060618.0,
            "endDOY": 20300101.0,
            "LAT": 51.55,
            "LON": 5.55,
            "timezn": 1.0,
            # Angles
            "tts": 35.0,
            "tto": 0.0,
            "psi": 0.0,
        }

    def __enter__(self):
        """Start MATLAB engine and add SCOPE path."""
        self.eng = matlab.engine.start_matlab("-nodisplay")
        self.eng.addpath(self.eng.genpath(str(self.scope_dir)), nargout=0)
        self.eng.cd(str(self.scope_dir), nargout=0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop MATLAB engine and restore input file."""
        if self.eng:
            self.eng.quit()

        if hasattr(self, "temp_backup") and self.temp_backup.exists():
            try:
                if self.original_input.exists():
                    os.remove(str(self.original_input))
                shutil.move(str(self.temp_backup), str(self.original_input))
            except Exception as e:
                print(f"Error restoring original input file: {e}")

    def _prepare_input(self, csv_content):
        """Write input directly to SCOPE's input directory, backing up the original."""
        input_dir = self.scope_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        temp_file = input_dir / "input_data.csv"

        # Backup original file if it exists
        if self.original_input.exists():
            self.temp_backup = self.original_input.with_name("input_data.backup.csv")
            if self.temp_backup.exists():
                os.remove(str(self.temp_backup))
            shutil.move(str(self.original_input), str(self.temp_backup))

        # Write new content
        with open(temp_file, "w") as f:
            f.write(csv_content)
        return temp_file

    def _generate_csv_content(self, params):
        """Generate CSV content with parameters in SCOPE's expected row-based format"""
        sections = [
            (
                "PROSPECT",
                [
                    "Cab",
                    "Cca",
                    "Cdm",
                    "Cw",
                    "Cs",
                    "Cant",
                    "Cp",
                    "Cbc",
                    "N",
                    "rho_thermal",
                    "tau_thermal",
                ],
            ),
            (
                "Leaf_Biochemical",
                [
                    "Vcmax25",
                    "BallBerrySlope",
                    "BallBerry0",
                    "Type",
                    "kV",
                    "Rdparam",
                    "Kn0",
                    "Knalpha",
                    "Knbeta",
                ],
            ),
            (
                "Leaf_Biochemical_magnani",
                ["Tyear", "beta", "kNPQs", "qLs", "stressfactor"],
            ),
            ("Fluorescence", ["fqe"]),
            (
                "Soil",
                [
                    "spectrum",
                    "rss",
                    "rs_thermal",
                    "cs",
                    "rhos",
                    "lambdas",
                    "SMC",
                    "BSMBrightness",
                    "BSMlat",
                    "BSMlon",
                ],
            ),
            (
                "Canopy",
                ["LAI", "hc", "LIDFa", "LIDFb", "leafwidth", "Cv", "crowndiameter"],
            ),
            ("Meteo", ["z", "Rin", "Ta", "Rli", "p", "ea", "u", "Ca", "Oa"]),
            (
                "Aerodynamic",
                [
                    "zo",
                    "d",
                    "Cd",
                    "rb",
                    "CR",
                    "CD1",
                    "Psicor",
                    "CSSOIL",
                    "rbs",
                    "rwc",
                ],
            ),
            ("timeseries", ["startDOY", "endDOY", "LAT", "LON", "timezn"]),
            ("Angles", ["tts", "tto", "psi"]),
        ]

        lines = []
        self.structured_params = defaultdict(dict)
        
        for section_name, param_names in sections:
            lines.append(f"{section_name},")
            for param in param_names:
                val = params.get(param, self.default_params.get(param, 0.0))
                if not isinstance(val, (list, tuple)):
                    val = [val]
                # Store with section key (e.g., "PROSPECT.Cab": [30,40,50])
                self.structured_params[section_name][param] = val
                line = f"{param}," + ",".join(map(str, val))
                lines.append(line)
            lines.append(",")
        return "\r\n".join(lines)

    def _find_latest_output(self):
        """Find the most recent output directory with enhanced logging."""
        output_parent = self.scope_dir / "output"
        if not output_parent.exists():
            raise FileNotFoundError(f"Output directory {output_parent} not found.")

        runs = list(output_parent.glob("scope_data_*"))
        if not runs:
            raise FileNotFoundError("No SCOPE output directories found.")

        # Debugging: Print found directories
        print("Found output directories:")
        for run in runs:
            print(f" - {run.name}")

        latest_run = max(runs, key=os.path.getmtime)
        print(f"Selected latest output: {latest_run}")
        return latest_run

    def _read_data(self, filepath):
        """Reads numerical data from CSV, TXT, or BIN files."""
        if filepath.suffix.lower() == ".csv":
            try:
                # All spectral files: no headers, skip comments
                if any(x in filepath.name for x in ["spectrum", "Esun", "Esky", "Eout", "fluorescence", "sigmaF"]):
                    df = pd.read_csv(filepath, sep=",", header=None, comment='#')
                    # Convert to list of lists, where each sublist is one simulation's spectrum
                    return df.values.tolist()  # Now returns [n_simulations, n_wavelengths]
                # Scalar files: handle headers and comments
                else:
                    df = pd.read_csv(filepath, sep=",", header=0, comment='#')
                    return df.values.tolist()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return None

        elif filepath.suffix.lower() == ".txt":
            with open(filepath, "r") as f:
                # Special handling for wavelength files
                if filepath.name in ["wlS.txt", "wlF.txt"]:
                    # Read all values into a single flat list
                    values = []
                    for line in f:
                        line_values = [float(v) for v in line.strip().split()]
                        values.extend(line_values)
                    return [values]  # Wrap in list to match other formats
                
                # Regular TXT file handling
                return [[float(value) for value in re.split(r'\s+', line.strip())] 
                       for line in f]
            
        elif filepath.suffix.lower() == ".bin":
            try:
                with open(filepath, "rb") as f:
                    # Assuming 4-byte floats (float32)
                    buffer = f.read()
                    num_floats = len(buffer) // 4
                    data = struct.unpack(f"{num_floats}f", buffer)
                    return [list(data)] #Consistent output
            except Exception as e:
                print(f"Error reading BIN file {filepath}: {e}")
                return None

        else:
            print(f"Unsupported file type: {filepath.suffix}")
            return None

    def _read_parameters(self, filepath):
        """Reads parameters from CSV files, handling multiple values."""
        params = {}
        if filepath.suffix.lower() == ".csv":
            if "filenames" in filepath.name:
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) >= 2:
                            params[row[0].strip()] = row[1].strip()
            else:
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    current_section = None
                    for row in reader:
                        if not row:
                            continue
                        if row[0].endswith(","):
                            current_section = row[0].strip(",")
                        elif len(row) >= 1 and current_section:
                            param_name = row[0].strip()
                            values = [cell.strip() for cell in row[1:] if cell.strip()]
                            converted_values = []
                            for val in values:
                                try:
                                    converted_values.append(float(val))
                                except ValueError:
                                    converted_values.append(val)
                            if len(converted_values) == 1:
                                param_value = converted_values[0]
                            else:
                                param_value = converted_values
                            key = f"{current_section}.{param_name}"
                            params[key] = param_value
        elif filepath.suffix.lower() == ".txt":
            try:
                with open(filepath, 'r') as f:
                    params[filepath.stem] = f.read().strip()
            except Exception as e:
                print(f"Error reading TXT file {filepath}: {e}")
        return params

    def generate_results_json(self, output_dir: Path) -> Path:
        """Generate results.json with properly formatted parameters."""
        results = {"scalar_outputs": {}}
        
        param_mapping = {}
        for section, params in self.structured_params.items():
            for param in params.keys():
                param_mapping[param] = f"{section}.{param}"

        param_file = output_dir / "pars_and_input_short.csv"
        if param_file.exists():
            param_df = pd.read_csv(param_file, comment='#')
            
            param_df = param_df.loc[:, ~param_df.columns.str.contains('^Unnamed')]
            param_df = param_df.drop(columns=['n_pars'], errors='ignore')
            
            param_df = param_df.rename(columns=param_mapping)
            
            run_parameters = []
            for _, row in param_df.iterrows():
                param_dict = {}
                for col in param_df.columns:
                    try:
                        param_dict[col] = float(row[col])
                    except (ValueError, TypeError):
                        param_dict[col] = row[col]
                run_parameters.append(param_dict)
            
            fixed_params = {
                f"{section}.{param}": vals[0]  # Get scalar value
                for section, params in self.structured_params.items()
                for param, vals in params.items()
                if f"{section}.{param}" not in param_df.columns
            }
            
            for param_dict in run_parameters:
                param_dict.update(fixed_params)
            
            results.update({
                "run_parameters": run_parameters,
                "num_simulations": len(run_parameters),
                "input_parameters": self.structured_params,
                "setoptions": self.active_setoptions
            })
        else:
            results.update({
                "num_simulations": 0,
                "input_parameters": self.structured_params,
                "setoptions": self.active_setoptions
            })

        results["scalar_outputs"] = {}
        for key, filename in self.scalar_files.items():
            filepath = output_dir / filename
            if filepath.exists():
                data = self._read_data(filepath)
                if data is not None:
                    results["scalar_outputs"][key] = data

        if self.setoptions.get("saveCSV", 1) == 0:  # Default to 1 if not present.
            bin_files = {
                "Rin": "Rin.bin",
                "Rli": "Rli.bin",
                "fluorescence_bin": "fluorescence.bin",  
            }
            for key, filename in bin_files.items():
                filepath = output_dir / filename
                if filepath.exists():
                    data = self._read_data(filepath)
                    if data is not None:
                        results["scalar_outputs"][key] = data

        # Process Parameters directory, skip filenames
        params_dir = output_dir / "Parameters"
        if params_dir.exists() and params_dir.is_dir():
            for param_file in params_dir.glob("*"):
                if param_file.is_file():
                    # Skip filenames files
                    if param_file.name.startswith("filenames"):
                        continue
                    params = self._read_parameters(param_file)
                    results.setdefault("model_parameters", {}).update(params)

        results["spectral_outputs"] = {}
        for key, filename in self.spectral_files.items():
            filepath = output_dir / filename
            if filepath.exists():
                data = self._read_data(filepath)
                if data is not None:
                    results["spectral_outputs"][key] = data



        # Add wlF and wlS
        wlf_file = output_dir / "wlF.txt"
        wls_file = output_dir / "wlS.txt"
        if wlf_file.exists():
            results["wlF"] = self._read_data(wlf_file)
        if wls_file.exists():
            results["wlS"] = self._read_data(wls_file)



        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        return json_path

    def run(self, params, setoptions=None):
        """Run SCOPE, return results.json path.  Includes debugging prints."""

        try:
            if setoptions:
                self._update_setoptions(setoptions)  # Update setoptions.csv
            # Debugging: Print parameters before CSV generation
            print("=== Parameters before CSV generation ===")
            print(json.dumps(params, indent=2))

            # Prepare and write input data
            csv_content = self._generate_csv_content(params)
            print("\n=== Generated CSV Content ===")
            print(csv_content)  # Print the generated CSV content

            input_file = self._prepare_input(csv_content)

            # Run SCOPE
            self.eng.run_scope_wrapper_json(nargout=0)

            # Find output and generate results.json
            output_dir = self._find_latest_output()
            json_path = self.generate_results_json(output_dir)

            return json_path

        except matlab.engine.MatlabExecutionError as e:
            print(f"MATLAB Error: {str(e)}")
            raise
        except FileNotFoundError as e:
            print(f"File Not Found Error: {str(e)}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

    def _update_setoptions(self, user_options):
        """Update setoptions while maintaining SCOPE's order and user values."""
        # Define required order of keys (without hardcoding values)
        required_keys = [
            "lite",
            "calc_fluor",
            "calc_planck",
            "calc_xanthophyllabs",
            "soilspectrum",
            "Fluorescence_model",
            "applTcorr",
            "verify",
            "saveCSV",
            "mSCOPE",
            "simulation",
            "calc_directional",
            "calc_vert_profiles",
            "soil_heat_method",
            "calc_rss_rbs",
            "MoninObukhov",
            "save_spectral"
        ]

        # Merge user options with defaults, prioritizing user values
        combined_options = {**self.active_setoptions, **user_options}
        
        # Build ordered list using required key order
        ordered_options = []
        for key in required_keys:
            value = combined_options.get(key, 0)  # Default to 0 if not provided
            ordered_options.append((str(value), key))

        setoptions_path = self.scope_dir / "input" / "setoptions.csv"
        with open(setoptions_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ordered_options)
        
        self.active_setoptions = combined_options

class SCOPEWrapper:
    def __init__(self):
        self.base_dir = Path.cwd()  
        self.scope_dir = self.base_dir / "SCOPE"
        self.original_input = self.scope_dir / "input" / "input_data.csv"
        self.default_params = self._get_default_parameters()
        self.eng = None 

    def _get_default_parameters(self):
        """Comprehensive default parameters for SCOPE """
        return {
            # PROSPECT
            'Cab': 40.0, 'Cca': 10.0, 'Cdm': 0.012, 'Cw': 0.009,
            'Cs': 0.0, 'Cant': 1.0, 'Cp': 0.0, 'Cbc': 0.0,
            'N': 1.5, 'rho_thermal': 0.01, 'tau_thermal': 0.01,
            
            # Leaf Biochemical
            'Vcmax25': 60.0, 'BallBerrySlope': 8.0, 'BallBerry0': 0.01,
            'Type': 0.0, 'kV': 0.64, 'Rdparam': 0.015, 'Kn0': 2.48,
            'Knalpha': 2.83, 'Knbeta': 0.114,
            
            # Magnani
            'Tyear': 15.0, 'beta': 0.51, 'kNPQs': 0.0, 
            'qLs': 1.0, 'stressfactor': 1.0,
            
            # Fluorescence
            'fqe': 0.01,
            
            # Soil
            'spectrum': 1.0, 'rss': 500.0, 'rs_thermal': 0.06,
            'cs': 1180.0, 'rhos': 1800.0, 'lambdas': 1.55,
            'SMC': 25.0, 'BSMBrightness': 0.5, 'BSMlat': 25.0,
            'BSMlon': 45.0,
            
            # Canopy
            'LAI': 3.0, 'hc': 2.0, 'LIDFa': -0.35, 'LIDFb': -0.15,
            'leafwidth': 0.1, 'Cv': 1.0, 'crowndiameter': 1.0,
            
            # Meteo
            'z': 5.0, 'Rin': 800.0, 'Ta': 20.0, 'Rli': 300.0,
            'p': 970.0, 'ea': 15.0, 'u': 2.0, 'Ca': 410.0, 'Oa': 209.0,
            
            # Aerodynamic
            'zo': 0.25, 'd': 1.34, 'Cd': 0.3, 'rb': 10.0,
            'CR': 0.35, 'CD1': 20.6, 'Psicor': 0.2,
            'CSSOIL': 0.01, 'rbs': 10.0, 'rwc': 0.0,
            
            # Timeseries
            'startDOY': 20060618.0, 'endDOY': 20300101.0,
            'LAT': 51.55, 'LON': 5.55, 'timezn': 1.0,
            
            # Angles
            'tts': 35.0, 'tto': 0.0, 'psi': 0.0
        }

    def __enter__(self):
        """Start MATLAB engine and add SCOPE path."""
        self.eng = matlab.engine.start_matlab('-nodisplay')  # -nodisplay for no GUI
        self.eng.addpath(self.eng.genpath(str(self.scope_dir)), nargout=0)  # Use genpath
        self.eng.cd(str(self.scope_dir), nargout=0)  # Change to SCOPE directory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop MATLAB engine and restore input file."""
        if self.eng:
            self.eng.quit()

         # Restore original file (important for repeated runs)
        if hasattr(self, 'temp_backup') and self.temp_backup.exists():
            try:
                if self.original_input.exists():
                    os.remove(str(self.original_input))
                shutil.move(str(self.temp_backup), str(self.original_input))
            except Exception as e:
                print(f"Error restoring original input file: {e}")



    def _prepare_input(self, csv_content):
        """Write input directly to SCOPE's input directory"""
        input_dir = self.scope_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True) 
        temp_file = input_dir / "input_data.csv"

        # Backup original file if it exists
        if self.original_input.exists():
            self.temp_backup = self.original_input.with_name("input_data.backup.csv")
            if self.temp_backup.exists():
                os.remove(str(self.temp_backup)) # Delete previous backup to avoid errors.
            shutil.move(str(temp_file), str(self.temp_backup))

        # Write new content
        with open(temp_file, 'w') as f:
            f.write(csv_content)
        return temp_file
            

    def _generate_csv_content(self, params):
        """Generate CSV content with exact SCOPE formatting (same as before)."""
        sections = [
            ('PROSPECT', [
                'Cab', 'Cca', 'Cdm', 'Cw', 'Cs', 'Cant', 
                'Cp', 'Cbc', 'N', 'rho_thermal', 'tau_thermal'
            ]),
            ('Leaf_Biochemical', [
                'Vcmax25', 'BallBerrySlope', 'BallBerry0', 'Type',
                'kV', 'Rdparam', 'Kn0', 'Knalpha', 'Knbeta'
            ]),
            ('Leaf_Biochemical_magnani', [
                'Tyear', 'beta', 'kNPQs', 'qLs', 'stressfactor'
            ]),
            ('Fluorescence', ['fqe']),
            ('Soil', [
                'spectrum', 'rss', 'rs_thermal', 'cs', 'rhos',
                'lambdas', 'SMC', 'BSMBrightness', 'BSMlat', 'BSMlon'
            ]),
            ('Canopy', [
                'LAI', 'hc', 'LIDFa', 'LIDFb', 'leafwidth',
                'Cv', 'crowndiameter'
            ]),
            ('Meteo', [
                'z', 'Rin', 'Ta', 'Rli', 'p', 'ea', 'u', 'Ca', 'Oa'
            ]),
            ('Aerodynamic', [
                'zo', 'd', 'Cd', 'rb', 'CR', 'CD1', 'Psicor',
                'CSSOIL', 'rbs', 'rwc'
            ]),
            ('timeseries', [
                'startDOY', 'endDOY', 'LAT', 'LON', 'timezn'
            ]),
            ('Angles', ['tts', 'tto', 'psi'])
        ]
        content = []
        for section, parameters in sections:
            content.append(f"{section},")
            for param in parameters:
                value = params.get(param, self.default_params[param])  # Use get with default
                content.append(f"{param},{float(value)}")
            content.append(",")
            
        return "\r\n".join(content) + "\r\n"



    def _find_latest_output(self):
        """Find the newest output directory (same as before)."""
        output_parent = self.scope_dir / "output"
        runs = sorted(output_parent.glob("example_run_*"), key=os.path.getmtime, reverse=True)
        if not runs:
            raise FileNotFoundError("No SCOPE output directories found.")
        return runs[0]

    def _read_data(self, filepath):
        """Reads data from a file, handling both CSV and TXT."""
        if filepath.suffix.lower() == '.csv':
            try:
                return np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=float).tolist()
            except Exception as e:
                print(f"Error reading CSV file {filepath}: {e}")
                return None  # Or [] or {} depending on context
        elif filepath.suffix.lower() == '.txt':
            try:
                wavelengths = []
                with open(filepath, 'r') as f:
                    for line in f:
                        # Split each line by spaces and convert to floats
                        values = [float(value) for value in line.split()]
                        wavelengths.extend(values)
                return wavelengths
            except Exception as e:
                print(f"Error reading TXT file {filepath}: {e}")
                return None
        else:
            print(f"Unsupported file type: {filepath.suffix}")
            return None
    
    
    def _read_parameters(self, filepath):
        """Reads parameters from a CSV file, handling section headers."""
        params = {}
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            current_section = None
            for row in reader:
                if not row:  # Skip empty lines
                    continue
                if row[0].endswith(','):  # Section header
                    current_section = row[0].strip(',')
                elif len(row) >= 2 and current_section:
                    param_name = row[0].strip()
                    param_value = row[1].strip()
                    # Convert to float if possible, otherwise keep as string
                    try:
                        param_value = float(param_value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
                    params[f"{current_section}.{param_name}"] = param_value
        return params


    def generate_results_json(self, output_dir: Path) -> Path:
        """Generate a comprehensive results.json from all SCOPE outputs."""
        results = {}
        # Process the Parameters subdirectory
        params_dir = output_dir / "Parameters"
        if params_dir.exists() and params_dir.is_dir():
            for param_file in params_dir.glob("*.csv"):
                if param_file.name.startswith("filenames"):
                    # Special handling for filenames
                    filenames = {}
                    with open(param_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2:
                                filenames[row[0].strip()] = row[1].strip()
                    results['filenames'] = filenames
                elif param_file.name.startswith("input_data"):
                    results['input_data'] = self._read_parameters(param_file)
                elif param_file.name.startswith("setoptions"):
                    results['setoptions'] = self._read_parameters(param_file)

        # Read data files (CSV and TXT) in the main output directory
        for file in output_dir.iterdir():
            if file.is_file():  # Only process files
                data = self._read_data(file)
                if data is not None:
                    results[file.stem] = data
        
        
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        return json_path

    def _write_scope_csv(self, params):
        """Write parameters directly to SCOPE's input_data.csv"""
        input_path = self.scope_dir / "input" / "input_data.csv"
        
        # Generate CSV content using your existing method
        csv_content = self._generate_csv_content(params)  
        
        # Write directly to SCOPE's expected input location
        with open(input_path, 'w', newline='') as f:
            f.write(csv_content)

    def run(self, params):
      """Run SCOPE with the given parameters and return parsed output."""
      try:
          self._write_scope_csv(params)
          self.eng.run_scope_wrapper_json(nargout=0)
          output_dir = self._find_latest_output()
          json_path = self.generate_results_json(output_dir)
          return json_path


      except matlab.engine.MatlabExecutionError as e:
          print(f"MATLAB Error: {str(e)}")
          raise 
      except FileNotFoundError as e:
          print(f"File Not Found Error: {str(e)}")
          raise
      except Exception as e:
          print(f"An unexpected error occurred: {str(e)}")
          raise