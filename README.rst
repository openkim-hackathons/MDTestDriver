NPT Crystal Structure
=====================

Example test driver that uses Lammps to run a molecular-dynamics simulation with a fixed number of atoms N at constant
temperature T and pressure P (NPT ensemble) to compute the equilibrium crystal structure.

This test driver repeats the unit cell to build a supercell and then runs a molecular-dynamics simulation in the
NPT ensemble using Lammps.

This test driver uses kim-convergence to detect an equilibrated molecular-dynamics simulation. It checks
convergence of the volume, temperature, enthalpy and cell shape parameters every 10000 timesteps.

During the equilibrated part of the simulation, the test driver averages the cell parameters and atomic positions to
obtain the equilibrium crystal structure. This includes an average over time, and an average over the replicated unit
cells.

After the molecular-dynamics simulation, the symmetry of the average structure is checked to ensure that it did not
change in comparison to the initial structure. Also, it is ensured that replicated atoms in replicated unit atoms are
not too far away from the average atomic positions.

The crystal might melt or vaporize during the simulation. In that case, kim-convergence would only detect
equilibration after an unnecessarily long simulation. Therefore, this test driver initially check for melting or
vaporization during a short initial simulation. During this initial run, the mean-squared displacement (MSD) of atoms
during the simulation is monitored. If the MSD exceeds a given threshold value, an error is raised.
