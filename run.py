import argparse
import subprocess
from kim_tools import query_crystal_structures
from test_driver.test_driver import TestDriver


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Pass arguments to the KIM test driver')
    parser.add_argument('-m', '--model', help='Pass model for test driver to run', required=True)
    parser.add_argument('-s', '--stoichiometry', help='Stoichiometry of structure to test', nargs='+',
                        required=True)
    parser.add_argument('-p', '--prototype', help='Prototype ASE label', required=True)
    args = parser.parse_args()

    # Get arguments
    model_name = args.model
    stoich = args.stoichiometry
    prot = args.prototype

    # Run test
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = TestDriver(model_name)
    list_of_queried_structures = query_crystal_structures(kim_model_name=model_name,
                                                          stoichiometric_species=stoich,
                                                          prototype_label=prot)
    for i, queried_structure in enumerate(list_of_queried_structures):
        test_driver(queried_structure, temperature_K=293.15,
                    cell_cauchy_stress_eV_angstrom3=[-6.241509074460762e-7, -6.241509074460762e-7, -6.241509074460762e-7,
                                                     0.0, 0.0, 0.0],
                    temperature_step_fraction=0.01, number_symmetric_temperature_steps=1, timestep=0.001,
                    number_sampling_timesteps=100, repeat=(0, 0, 0), max_workers=3)
        test_driver.write_property_instances_to_file(filename=f"output/results_{i}.edn")
