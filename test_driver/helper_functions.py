from math import ceil
import os
import random
import re
import subprocess
from typing import Iterable, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def run_lammps(modelname: str, temperature: float, pressure: float, timestep: float, number_sampling_timesteps: int,
               species: List[str], msd_threshold: float, lammps_command: str) -> Tuple[str, str, str, str] | None:
    # Get random 31-bit unsigned integer.
    seed = random.getrandbits(31)

    pdamp = timestep * 100.0
    tdamp = timestep * 1000.0

    log_filename = "output/lammps.log"
    restart_filename = "output/final_configuration.restart"
    variables = {
        "modelname": modelname,
        "temperature": temperature,
        "temperature_seed": seed,
        "temperature_damping": tdamp,
        "pressure": pressure,
        "pressure_damping": pdamp,
        "timestep": timestep,
        "number_sampling_timesteps": number_sampling_timesteps,
        "species": " ".join(species),
        "average_position_filename": "output/average_position.dump.*",
        "average_cell_filename": "output/average_cell.dump",
        "write_restart_filename": restart_filename,
        "trajectory_filename": "output/trajectory.lammpstrj",
        "msd_threshold": msd_threshold
    }

    command = (
            f"{lammps_command} "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + f" -log {log_filename}"
            + " -in npt.lammps")

    subprocess.run(command, check=True, shell=True)

    plot_property_from_lammps_log(log_filename, ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))

    equilibration_time = extract_equilibration_step_from_logfile(log_filename)
    # Round to next multiple of 10000.
    equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000

    full_average_position_file = "output/average_position.dump.full"
    compute_average_positions_from_lammps_dump("output",
                                               "average_position.dump",
                                               full_average_position_file, equilibration_time)

    full_average_cell_file = "output/average_cell.dump.full"
    compute_average_cell_from_lammps_dump("output/average_cell.dump",
                                          full_average_cell_file, equilibration_time)

    return log_filename, restart_filename, full_average_position_file, full_average_cell_file


def plot_property_from_lammps_log(in_file_path: str, property_names: Iterable[str]) -> None:
    """
    The function to get the value of the property with time from ***.log
    the extracted data are stored as ***.csv and ploted as property_name.png
    data_dir --- the directory contains lammps_equilibration.log
    property_names --- the list of properties
    """

    def get_table(in_file):
        if not os.path.isfile(in_file):
            raise FileNotFoundError(in_file + " not found")
        elif ".log" not in in_file:
            raise FileNotFoundError("The file is not a *.log file")
        is_first_header = True
        header_flags = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
        eot_flags = ["Loop", "time", "on", "procs", "for", "steps"]
        table = []
        with open(in_file, "r") as f:
            line = f.readline()
            while line:  # Not EOF.
                is_header = True
                for _s in header_flags:
                    is_header = is_header and (_s in line)
                if is_header:
                    if is_first_header:
                        table.append(line)
                        is_first_header = False
                    content = f.readline()
                    while content:
                        is_eot = True
                        for _s in eot_flags:
                            is_eot = is_eot and (_s in content)
                        if not is_eot:
                            table.append(content)
                        else:
                            break
                        content = f.readline()
                line = f.readline()
        return table

    def write_table(table, out_file):
        with open(out_file, "w") as f:
            for l in table:
                f.writelines(l)

    dir_name = os.path.dirname(in_file_path)
    in_file_name = os.path.basename(in_file_path)
    out_file_path = os.path.join(dir_name, in_file_name.replace(".log", ".csv"))

    table = get_table(in_file_path)
    write_table(table, out_file_path)
    df = np.loadtxt(out_file_path, skiprows=1, usecols=tuple(range(16)))

    for property_name in property_names:
        with open(out_file_path) as file:
            first_line = file.readline().strip("\n")
        property_index = first_line.split().index(property_name)
        properties = df[:, property_index]
        step = df[:, 0]
        plt.plot(step, properties)
        plt.xlabel("step")
        plt.ylabel(property_name)
        img_file = os.path.join(dir_name, in_file_name.replace(".log", "_") + property_name + ".png")
        plt.savefig(img_file, bbox_inches="tight")
        plt.close()


def extract_equilibration_step_from_logfile(filename: str) -> int:
    # Get file content.
    with open(filename, 'r') as file:
        data = file.read()

    # Look for pattern.
    exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
    mean_pattern = r'"equilibration_step"\s*([^ ]+)'
    match_init = re.search(exterior_pattern, data, re.DOTALL)
    equil_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
    if equil_matches is None:
        raise ValueError("Equilibration step not found")

    # Return largest match.
    return max(int(equil) for equil in equil_matches)


def compute_average_positions_from_lammps_dump(data_dir: str, file_str: str, output_filename: str,
                                               skip_steps: int) -> None:
    """
    This function compute the average position over *.dump files which contains the file_str in data_dir and output it
    to data_dir/[file_str]_over_dump.out

    input:
    data_dir -- the directory contains all the data e.g average_position.dump.* files
    file_str -- the files whose names contain the file_str are considered
    output_filename -- the name of the output file
    skip_steps -- dump files with steps <= skip_steps are ignored
    """

    def get_id_pos_dict(file_name):
        '''
        input:
        file_name--the file_name that contains average postion data
        output:
        the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
        for the averaged positions over files
        '''
        id_pos_dict = {}
        header4N = ["NUMBER OF ATOMS"]
        header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
        is_table_started = False
        is_natom_read = False
        with open(file_name, "r") as f:
            line = f.readline()
            count_content_line = 0
            N = 0
            while line:
                if not is_natom_read:
                    is_natom_read = np.all([flag in line for flag in header4N])
                    if is_natom_read:
                        line = f.readline()
                        N = int(line)
                if not is_table_started:
                    contain_flags = np.all([flag in line for flag in header4pos])
                    is_table_started = contain_flags
                else:
                    count_content_line += 1
                    words = line.split()
                    id = int(words[0])
                    pos = np.array([float(words[1]), float(words[2]), float(words[3])])
                    id_pos_dict[id] = pos
                if count_content_line > 0 and count_content_line >= N:
                    break
                line = f.readline()
        if count_content_line < N:
            print("The file " + file_name +
                  " is not complete, the number of atoms is smaller than " + str(N))
        return id_pos_dict

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir + " does not exist")
    if not ".dump" in file_str:
        raise ValueError("file_str must be a string containing .dump")

    # Extract and store all the data.
    pos_list = []
    max_step, last_step_file = -1, ""
    for file_name in os.listdir(data_dir):
        if file_str in file_name:
            step = int(re.findall(r'\d+', file_name)[-1])
            if step <= skip_steps:
                continue
            file_path = os.path.join(data_dir, file_name)
            id_pos_dict = get_id_pos_dict(file_path)
            id_pos = sorted(id_pos_dict.items())
            id_list = [pair[0] for pair in id_pos]
            pos_list.append([pair[1] for pair in id_pos])
            # Check if this is the last step.
            if step > max_step:
                last_step_file, max_step = os.path.join(data_dir, file_name), step
    if max_step == -1 and last_step_file == "":
        raise RuntimeError("Found no files to average over.")
    pos_arr = np.array(pos_list)
    avg_pos = np.mean(pos_arr, axis=0)
    # Get the lines above the table from the file of the last step.
    with open(last_step_file, "r") as f:
        header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
        line = f.readline()
        description_str = ""
        is_table_started = False
        while line:
            description_str += line
            is_table_started = np.all([flag in line for flag in header4pos])
            if is_table_started:
                break
            else:
                line = f.readline()
    # Write the output to the file.
    with open(output_filename, "w") as f:
        f.write(description_str)
        for i in range(len(id_list)):
            f.write(str(id_list[i]))
            f.write("  ")
            for dim in range(3):
                f.write('{:3.6}'.format(avg_pos[i, dim]))
                f.write("  ")
            f.write("\n")


def compute_average_cell_from_lammps_dump(input_file: str, output_file: str, skip_steps: int) -> None:
    with open(input_file, "r") as f:
        f.readline()  # Skip the first line.
        header = f.readline()
        header = header.replace("#", "")
    property_names = header.split()
    data = np.loadtxt(input_file, skiprows=2)
    time_step_index = property_names.index("TimeStep")
    time_step_data = data[:, time_step_index]
    cutoff_index = np.argmax(time_step_data > skip_steps)
    assert time_step_data[cutoff_index] > skip_steps
    assert cutoff_index == 0 or time_step_data[cutoff_index - 1] <= skip_steps
    mean_data = data[cutoff_index:].mean(axis=0).tolist()
    with open(output_file, "w") as f:
        print("# Full time-averaged data for cell information", file=f)
        print(f"# {' '.join(name for name in property_names if name != 'TimeStep')}", file=f)
        print(" ".join(str(mean_data[i]) for i, name in enumerate(property_names) if name != "TimeStep"), file=f)


def get_positions_from_averaged_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
    lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
    return [(line[1], line[2], line[3]) for line in lines]


def get_cell_from_averaged_lammps_dump(filename: str) -> npt.NDArray[np.float64]:
    cell_list = np.loadtxt(filename, comments='#')
    assert len(cell_list) == 6
    cell = np.empty(shape=(3, 3))
    cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
    cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
    cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
    return cell


def check_lammps_log_for_wrong_structure_format(log_file):
    wrong_format_in_structure_file = False

    try:
        with open(log_file, "r") as logfile:
            data = logfile.read()
            data = data.split("\n")
            final_line = data[-2]

            if final_line == "Last command: read_data output/zero_temperature_crystal.lmp":
                wrong_format_in_structure_file = True
    except FileNotFoundError:
        pass

    return wrong_format_in_structure_file
