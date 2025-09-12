import os
import shutil
import subprocess
from typing import Sequence
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.lammps import convert, Prism
import numpy as np
from kim_tools import KIMTestDriverError
from kim_tools.symmetry_util.core import reduce_and_avg, kstest_reduced_distances, PeriodExtensionException
from kim_tools.test_driver import SingleCrystalTestDriver
from .helper_functions import (check_lammps_log_for_wrong_structure_format,
                               get_cell_from_averaged_lammps_dump, get_positions_from_averaged_lammps_dump, run_lammps)


class TestDriver(SingleCrystalTestDriver):
    def _calculate(self, timestep: float, number_sampling_timesteps: int = 100, repeat: Sequence[int] = (3, 3, 3),
                   lammps_command = "lmp", msd_threshold: float = 0.1, **kwargs) -> None:
        """
        Compute crystal structure at constant pressure and temperature (NPT).
        """
        # Set prototype label
        self.prototype_label = self._get_nominal_crystal_structure_npt()["prototype-label"]["source-value"]

        # Get temperature in Kelvin.
        temperature_K = self._get_temperature(unit="K")

        # Get cauchy stress tensor in bar.
        cell_cauchy_stress_bar = self._get_cell_cauchy_stress(unit="bar")

        # Check arguments.
        if not temperature_K > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")

        if not len(cell_cauchy_stress_bar) == 6:
            raise RuntimeError("Specify all six (x, y, z, xy, xz, yz) entries of the cauchy stress tensor.")

        if not (cell_cauchy_stress_bar[0] == cell_cauchy_stress_bar[1] == cell_cauchy_stress_bar[2]):
            raise RuntimeError("The diagonal entries of the stress tensor have to be equal so that a hydrostatic "
                               "pressure is used.")

        if not (cell_cauchy_stress_bar[3] == cell_cauchy_stress_bar[4] == cell_cauchy_stress_bar[5] == 0.0):
            raise RuntimeError("The off-diagonal entries of the stress tensor have to be zero so that a hydrostatic "
                               "pressure is used.")

        if not number_sampling_timesteps > 0:
            raise RuntimeError("Number of timesteps between sampling in Lammps has to be bigger than zero.")

        if not len(repeat) == 3:
            raise RuntimeError("The repeat argument has to be a tuple of three integers.")

        if not all(r >= 0 for r in repeat):
            raise RuntimeError("All number of repeats must be bigger than zero.")

        if not msd_threshold > 0.0:
            raise RuntimeError("The mean-squared displacement threshold has to be bigger than zero.")

        # Get pressure from cauchy stress tensor.
        pressure_bar = -cell_cauchy_stress_bar[0]
        
        # Copy original atoms so that their information does not get lost.
        original_atoms = self._get_atoms().copy()

        # Create atoms object that will contain the supercell.
        atoms_new = self._get_atoms().copy()

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        # Build supercell.
        if repeat == (0, 0, 0):
            # Get a size close to 10K atoms (shown to give good convergence)
            x = int(np.ceil(np.cbrt(10000 / len(atoms_new))))
            repeat = (x, x, x)

        atoms_new = atoms_new.repeat(repeat)

        # Create output directory for all data files and copy over necessary files.
        os.makedirs("output", exist_ok=True)
        test_driver_directory = os.path.dirname(os.path.realpath(__file__))
        if os.getcwd() != test_driver_directory:
            shutil.copyfile(os.path.join(test_driver_directory, "npt.lammps"), "npt.lammps")
            shutil.copyfile(os.path.join(test_driver_directory, "file_read_test.lammps"), "file_read_test.lammps")
            shutil.copyfile(os.path.join(test_driver_directory, "run_length_control.py"), "run_length_control.py")
        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        with open("accuracies.py", "w") as file:
            print("""from typing import Optional, Sequence

# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
# For cells, we can only use a relative accuracy for all non-zero variables.
# The last three variables, however, correspond to the tilt factors of the orthogonal cell (see npt.lammps which are
# expected to fluctuate around zero. For these, we should use an absolute accuracy instead.""", file=file)
            relative_accuracies = ["0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01"]
            absolute_accuracies = ["None", "None", "None", "None", "None", "None", "None", "None", "None"]
            _, _, _, xy, xz, yz = convert(Prism(atoms_new.get_cell()).get_lammps_prism(), "distance",
                                          "ASE", "metal")
            if abs(xy) < 1.0e-6:
                relative_accuracies[6] = "None"
                absolute_accuracies[6] = "0.01"
            if abs(xz) < 1.0e-6:
                relative_accuracies[7] = "None"
                absolute_accuracies[7] = "0.01"
            if abs(yz) < 1.0e-6:
                relative_accuracies[8] = "None"
                absolute_accuracies[8] = "0.01"
            print(f"RELATIVE_ACCURACY: Sequence[Optional[float]] = [{', '.join(relative_accuracies)}]", file=file)
            print(f"ABSOLUTE_ACCURACY: Sequence[Optional[float]] = [{', '.join(absolute_accuracies)}]", file=file)

        # Write lammps file.
        structure_file = "output/zero_temperature_crystal.lmp"
        atoms_new.write(structure_file, format="lammps-data", masses=True, units="metal")

        # Handle cases where kim models expect different structure file formats.
        try:
            run_lammps(self.kim_model_name, temperature_K, pressure_bar, timestep, number_sampling_timesteps, species,
                       msd_threshold, lammps_command=lammps_command, test_file_read=True)
        except subprocess.CalledProcessError as e:
            wrong_format_error = check_lammps_log_for_wrong_structure_format(
                "output/lammps_file_format_test_temperature_0.log")

            if wrong_format_error:
                # write the atom configuration file in the 'charge' format some models expect
                #assign_charges(atoms_new)
                write_lammps_data(structure_file, atoms_new, atom_style="charge", masses=True, units="metal")
                # try to read the file again, raise any exeptions that might happen
                run_lammps(self.kim_model_name, temperature_K, pressure_bar, timestep, number_sampling_timesteps,
                           species, msd_threshold, lammps_command=lammps_command, test_file_read=True)
            else:
                raise e

        # Run single Lammps simulation.
        log_filename, restart_filename, average_position_filename, average_cell_filename = run_lammps(
            self.kim_model_name, temperature_K, pressure_bar, timestep, number_sampling_timesteps, species,
            msd_threshold, lammps_command=lammps_command, test_file_read=False)

        # Cleanup.
        if os.getcwd() != test_driver_directory:
            os.remove("npt.lammps")
            os.remove("file_read_test.lammps")
            os.remove("run_length_control.py")

        # Check that crystal did not melt or vaporize.
        with open(log_filename, "r") as f:
            for line in f:
                if line.startswith("Crystal melted or vaporized"):
                    raise KIMTestDriverError(f"Crystal melted or vaporized during simulation at temperature {temperature_K} K.")

        # Process results and check that symmetry is unchanged after simulation.
        atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
        atoms_new.set_scaled_positions(
            get_positions_from_averaged_lammps_dump(average_position_filename))
        reduced_atoms, reduced_distances = reduce_and_avg(atoms_new, repeat)

        # Check that the symmetry of the structure did not change.
        if not self._verify_unchanged_symmetry(reduced_atoms):
            reduced_atoms.write(f"output/reduced_atoms_failing.poscar",
                                format="vasp", sort=True)
            raise KIMTestDriverError(f"Symmetry of structure changed during simulation at temperature {temperature_K} K.")

        # Check that the reduced distances are normally distributed.
        try:
            kstest_reduced_distances(reduced_distances, significance_level=0.05,
                                     plot_filename=f"output/reduced_distance_histogram.pdf",
                                     number_bins=20)
        except PeriodExtensionException as e:
            reduced_atoms.write(f"output/reduced_atoms_failing.poscar",
                                format="vasp", sort=True)
            raise KIMTestDriverError(f"Reduced distances are not normally distributed at temperature {temperature_K} K: {e}")

        # Write NPT crystal structure.
        self._update_nominal_parameter_values(reduced_atoms)
        self._add_property_instance_and_common_crystal_genome_keys("crystal-structure-npt", write_stress=True,
                                                                   write_temp=temperature_K)
        self._add_file_to_current_property_instance("restart-file",
                                                    "output/final_configuration.restart")

        print('####################################')
        print('# NPT Crystal Structure Results #')
        print('####################################')
        print(f'Temperature: {temperature_K} K')
        print(f'Pressure: {pressure_bar} bar')
