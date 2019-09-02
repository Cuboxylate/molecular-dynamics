# molecular-dynamics
Some simple molecular dynamics simulation scripts.

I was provided with 1dmd.py as part of a lab for BIOINF703, Semester 2, 2019 at the University of Auckland. I've extended on it to create 1dmd_boundary_conditions.py and 3dmd.py.

## 1dmd.py
This script produces a simple 1D molecular dynamics simulation output. It follows the steps:
1. Initialise the system:
   - Place particles evenly along a lone
   - Assign velocities at random, then scale them to reach desired initial temperature
2. During each timestep:
   - Calculate the force acting on each particle using Lennard-Jones potentials
   - Calculate the new position of each particle using this force and the Verlet Algorithm
   - Calculate the total temperature and potential energy of the system
   - Write all of these to appropriate files
3. After the simulation finishes:
   - Calculate and write the averages and standard deviations of temperature and energy

## 1dmd_boundary_conditions.py
This script is my first extension on 1dmd.py. The main purpose is to fix a bug around boundary condition handling, where the molecules would continue past the edge of the box until they collided. To fix this, I adjusted the final outputted coordinates to always be between 0 and 1 x box length.

I also cleaned up the script somewhat, and made each run output its files into a timestamped subfolder instead of overwriting the outputs each time. 

See an example output of this script in boundary_conditions_output. 

## 3dmd,py
In this script I extend 1DMD into three dimensions. The major changes I made to achieve this were:
1. Coordinates, velocities and forces are all now 3D objects, represented by numpy arrays. 
2. The initial molecule positions are now randomly placed in the 3D box, with no repetition.
3. The forces calculation needs to find the closest image of one molecule to another, through periodic boundaries, so I implemented a (not-very-efficient) search for it in `MD.find_closest_particle()`.
4. Modified the forces calculation to use the absolute distance to find the Lennard-Jones potential, then apply it to the distances between the molecules in each dimension separately.
5. Modified the temperature and potential energy calculations to take 3d into account. 

See an example output of this script in 3dmd_output, which used 150 molecules in a 22x22x22 box. 

