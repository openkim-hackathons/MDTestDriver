from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import shutil
import subprocess
from typing import Optional, Tuple
from ase.io.lammpsdata import write_lammps_data
import numpy as np
from kim_tools import get_stoich_reduced_list_from_prototype, query_crystal_genome_structures
from kim_tools.test_driver import CrystalGenomeTestDriver
from .helper_functions import (check_lammps_log_for_wrong_structure_format, compute_alpha, compute_heat_capacity,
                               get_cell_from_averaged_lammps_dump, get_positions_from_averaged_lammps_dump,
                               reduce_and_avg, run_lammps)


class HeatCapacity(CrystalGenomeTestDriver):
    def _calculate(self, temperature_step_fraction: float, number_symmetric_temperature_steps: int, timestep: float, 
                   number_sampling_timesteps: int = 100, repeat: Tuple[int, int, int] = (3, 3, 3), 
                   loose_triclinic_and_monoclinic: bool = False, max_workers: Optional[int] = None, **kwargs) -> None:
        """
        Compute constant-pressure heat capacity from centered finite difference (see Section 3.2 in
        https://pubs.acs.org/doi/10.1021/jp909762j).
        """
        # Check arguments.
        if not self.temperature_K > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")

        if not len(self.cell_cauchy_stress_eV_angstrom3) == 6:
            raise RuntimeError("Specify all six (x, y, z, xy, xz, yz) entries of the cauchy stress tensor.")
        
        if not self.cell_cauchy_stress_eV_angstrom3[0] == self.cell_cauchy_stress_eV_angstrom3[1] == self.cell_cauchy_stress_eV_angstrom3[2]:
            raise RuntimeError("The diagonal entries of the stress tensor have to be equal so that a hydrostatic pressure is used.")
        
        if not self.cell_cauchy_stress_eV_angstrom3[3] == self.cell_cauchy_stress_eV_angstrom3[4] == self.cell_cauchy_stress_eV_angstrom3[5]:
            raise RuntimeError("The off-diagonal entries of the stress tensor have to be zero so that a hydrostatic pressure is used.")
        
        if not number_symmetric_temperature_steps > 0:
            raise RuntimeError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise RuntimeError(
                "The given number of symmetric temperature steps and the given temperature-step fraction "
                "would yield zero or negative temperatures.")

        if not number_sampling_timesteps > 0:
            raise RuntimeError("Number of timesteps between sampling in Lammps has to be bigger than zero.")
        
        if not all(r > 0 for r in repeat):
            raise RuntimeError("All number of repeats must be bigger than zero.")

        if max_workers is not None and not max_workers > 0:
            raise RuntimeError("Maximum number of workers has to be bigger than zero.")
        
        # Convert stress to bar for Lammps using metal units.
        ev_angstrom3_to_bar_conversion_factor = 1.602176634e6
        cell_cauchy_stress_bar = [s * ev_angstrom3_to_bar_conversion_factor for s in self.cell_cauchy_stress_eV_angstrom3]
        pressure_bar = cell_cauchy_stress_bar[0]
        
        # Copy original atoms so that their information does not get lost.
        original_atoms = self.atoms.copy()

        # Create atoms object that will contain the supercell.
        atoms_new = self.atoms.copy()

        # UNCOMMENT THIS TO TEST A TRICLINIC STRUCTURE!
        # atoms_new = bulk('Ar', 'fcc', a=5.248)

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        # Build supercell.
        atoms_new = atoms_new.repeat(repeat)

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * self.temperature_K
        temperatures = [self.temperature_K + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data", masses=True)

        # Handle cases where kim models expect different structure file formats.
        try:
            run_lammps(self.kim_model_name, 0, temperatures[0], pressure_bar, timestep,
                       number_sampling_timesteps, species, test_file_read=True)
        except subprocess.CalledProcessError as e:
            filename = "output/lammps_file_format_test_temperature_{temperature_index}.log"
            log_file = os.path.join(TDdirectory, filename)
            wrong_format_error = check_lammps_log_for_wrong_structure_format(log_file)

            if wrong_format_error:
                # write the atom configuration file in the in the 'charge' format some models expect
                write_lammps_data(structure_file, atoms_new, atom_style="charge", masses=True, units="metal")
                # try to read the file again, raise any exeptions that might happen
                run_lammps(self.kim_model_name, 0, temperatures[0], pressure_bar, timestep,
                           number_sampling_timesteps, species, test_file_read=True)
            else:
                raise e

        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        if atoms_new.get_cell().orthorhombic:
            shutil.copyfile(os.path.join(TDdirectory, "accuracies_orthogonal.py"), 
                            os.path.join(TDdirectory, "accuracies.py"))
        else:
            shutil.copyfile(os.path.join(TDdirectory, "accuracies_non_orthogonal.py"), 
                            os.path.join(TDdirectory, "accuracies.py"))

        # Run Lammps simulations in parallel.
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, t in enumerate(temperatures):
                futures.append(executor.submit(
                    run_lammps, self.kim_model_name, i, t, pressure_bar, timestep,
                    number_sampling_timesteps, species))

        # If one simulation fails, cancel all runs.
        for future in as_completed(futures):
            assert future.done()
            exception = future.exception()
            if exception is not None:
                for f in futures:
                    f.cancel()
                raise exception

        # Collect results and check that symmetry is unchanged after all simulations.
        log_filenames = []
        restart_filenames = []
        middle_temperature_atoms = None
        middle_temperature = None
        for t_index, (future, t) in enumerate(zip(futures, temperatures)):
            assert future.done()
            assert future.exception() is None
            log_filename, restart_filename, average_position_filename, average_cell_filename = future.result()
            log_filenames.append(log_filename)
            restart_filenames.append(restart_filename)
            restart_filenames.append(restart_filename)
            atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
            atoms_new.set_scaled_positions(
                get_positions_from_averaged_lammps_dump(average_position_filename))
            reduced_atoms = reduce_and_avg(atoms_new, repeat)

            if t_index == number_symmetric_temperature_steps:
                # Store the atoms of the middle temperature for later because their crystal genome designation 
                # will be used for the heat-capacity and thermal expansion properties.
                middle_temperature_atoms = reduced_atoms.copy()
                middle_temperature = t
            self._update_crystal_genome_designation_from_atoms(
                reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
            self.temperature_K = t
            self._add_property_instance_and_common_crystal_genome_keys("crystal-structure-npt", write_stress=True, write_temp=True)
            self._add_file_to_current_property_instance("restart-file", 
                                                        os.path.join(TDdirectory, f"output/final_configuration_temperature_{t_index}.restart"))
            # Reset to original atoms.
            self._update_crystal_genome_designation_from_atoms(
                original_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
        assert middle_temperature_atoms is not None
        assert middle_temperature is not None

        c = compute_heat_capacity(temperatures, log_filenames, 2)
        alpha = compute_alpha(log_filenames, temperatures, self.prototype_label)

        # Print result.
        print('####################################')
        print('# NPT Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')
        print('####################################')
        print('# NPT Linear Thermal Expansion Tensor Results #')
        print('####################################')
        print(f'alpha:\t{alpha}')

        # Write property.
        max_accuracy = len(temperatures) - 1
        self._update_crystal_genome_designation_from_atoms(
                middle_temperature_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
        self.temperature_K = middle_temperature
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-npt", write_stress=True, write_temp=True)
        assert len(atoms_new) == len(original_atoms) * repeat[0] * repeat[1] * repeat[2]
        number_atoms = len(atoms_new)
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity_per_atom", 
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / number_atoms, 
            "eV/Kelvin",
            uncertainty_info={"source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / number_atoms})
        number_atoms_in_formula = sum(get_stoich_reduced_list_from_prototype(self.prototype_label))
        assert number_atoms % number_atoms_in_formula == 0
        number_formula = number_atoms // number_atoms_in_formula
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity_per_formula", 
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / number_formula, 
            "eV/Kelvin",
            uncertainty_info={"source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / number_formula})
        total_mass_g_per_mol = sum(atoms_new.get_masses())
        self._add_key_to_current_property_instance(
            "constant_pressure_specific_heat_capacity", 
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / total_mass_g_per_mol, 
            "eV/Kelvin/(g/mol)",
            uncertainty_info={"source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / total_mass_g_per_mol})

        alpha11 = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha11_err = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha12 = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha12_err = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha13 = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha13_err = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha22 = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha22_err = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha23 = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha23_err = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha33 = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha33_err = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][1]

        # enforce tensor symmetries
        alpha21 = alpha12
        alpha31 = alpha13
        alpha32 = alpha23

        alpha21_err = alpha12_err
        alpha31_err = alpha13_err
        alpha32_err = alpha23_err

        alpha_final = np.asarray([[alpha11, alpha12, alpha13],
                                  [alpha21, alpha22, alpha23],
                                  [alpha31, alpha32, alpha33]])

        alpha_final_err = np.asarray([[alpha11_err, alpha12_err, alpha13_err],
                                      [alpha21_err, alpha22_err, alpha23_err],
                                      [alpha31_err, alpha32_err, alpha33_err]])

        self._add_property_instance_and_common_crystal_genome_keys("thermal-expansion-coefficient-npt",
                                                                   write_stress=True, write_temp=True)
        space_group = int(self.prototype_label.split("_")[2])
        # alpha11 defined for all space groups
        self._add_key_to_current_property_instance("alpha11", alpha11, "1/K", uncertainty_info={"source-std-uncert-value":alpha11_err})

        alpha_symmetry_reduced = np.asarray([[alpha11, 0.0, 0.0],
                                             [0.0, alpha11, 0.0],
                                             [0.0, 0.0, alpha11 ]])
        
        alpha_symmetry_reduced_err = np.asarray([[alpha11_err, 0.0, 0.0],
                                                [0.0, alpha11_err, 0.0],
                                                [0.0, 0.0, alpha11_err ]])

        # hexagona, trigonal, tetragonal space groups also compute alpha33
        if space_group <= 194:
            self._add_key_to_current_property_instance("alpha33", alpha33, "1/K", uncertainty_info={"source-std-uncert-value":alpha33_err})

            alpha_symmetry_reduced = np.asarray([[alpha11, 0.0, 0.0],
                                                [0.0, alpha11, 0.0],
                                                [0.0, 0.0, alpha33 ]])
            
            alpha_symmetry_reduced_err = np.asarray([[alpha11_err, 0.0, 0.0],
                                                    [0.0, alpha11_err, 0.0],
                                                    [0.0, 0.0, alpha33_err ]])
        
        # orthorhombic, also compute alpha22
        if space_group <= 74:
            self._add_key_to_current_property_instance("alpha22", alpha22, "1/K", uncertainty_info={"source-std-uncert-value":alpha22_err})

            alpha_symmetry_reduced = np.asarray([[alpha11, 0.0, 0.0],
                                                [0.0, alpha22, 0.0],
                                                [0.0, 0.0, alpha33 ]])
            
            alpha_symmetry_reduced_err = np.asarray([[alpha11_err, 0.0, 0.0],
                                                    [0.0, alpha22_err, 0.0],
                                                    [0.0, 0.0, alpha33_err ]])
        
        # monoclinic or triclinic, compute all components
        if space_group <= 15:
            self._add_key_to_current_property_instance("alpha12", alpha12, "1/K", uncertainty_info={"source-std-uncert-value":alpha12_err})
            self._add_key_to_current_property_instance("alpha13", alpha13, "1/K", uncertainty_info={"source-std-uncert-value":alpha13_err})
            self._add_key_to_current_property_instance("alpha23", alpha23, "1/K", uncertainty_info={"source-std-uncert-value":alpha23_err})

            alpha_symmetry_reduced = alpha_final
            alpha_symmetry_reduced_err = alpha_final_err
        self._add_key_to_current_property_instance("thermal-expansion-tensor", alpha_final, "1/K", uncertainty_info={"source-std-uncert-value":alpha_final_err})
        self._add_key_to_current_property_instance("thermal-expansion-tensor-symmetry-reduced",alpha_symmetry_reduced,"1/K",uncertainty_info={"source-std-uncert-value":alpha_symmetry_reduced_err})


if __name__ == "__main__":
    model_name = "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacity(model_name)
    list_of_queried_structures = query_crystal_genome_structures(kim_model_name=model_name,
                                                                 stoichiometric_species=['Al'],
                                                                 prototype_label='A_cF4_225_a')
    for i, queried_structure in enumerate(list_of_queried_structures):
        test_driver(**queried_structure, temperature_K=293.15, 
                    cell_cauchy_stress_eV_angstrom3=[6.241509074460762e-7, 6.241509074460762e-7, 6.241509074460762e-7, 0.0, 0.0, 0.0], 
                    temperature_step_fraction=0.01, number_symmetric_temperature_steps=2, timestep=0.001, 
                    number_sampling_timesteps=100, repeat=(3, 3, 3), loose_triclinic_and_monoclinic=True, max_workers=5)
        test_driver.write_property_instances_to_file(filename=f"output/results_{i}.edn")
