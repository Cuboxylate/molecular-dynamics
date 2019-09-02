#!/usr/bin/python

"""
BIOINF703 Sem2 2019, Lab 2
Sebastian Dunn, sdun067

3DMD: Converting the 1D simulation to 3D

I extended this from 1dmd_bounday_condistions.py.

Major new changes:
    - Coordinates, forces and velocities are now
    numpy arrays, which act like vectors with normal
    operators, e.g +-*/
    - Original lattice positions are randomly chosen
    in the box now
    - Forces calculation:
        - Accounts for boundary conditions by
        finding the particle image j with shortest absolute
        distance to i
        - Compute Lennard-Jones potential using this
        absolute distance
        - Then apply it separately in each dimension
    - Integration calculation:
        - Converted naturally to 3D with minimal changes
        - sumv2 took some experimenting to get right, but
        I use the sum of each of the axis velocities squared
    - Some more renaming and tidy up

"""

# import needed modules
import math
from datetime import datetime
import os
import numpy as np
import random


def main(md):
    # Set up output files
    timestampString = datetime.now().strftime('%Y-%m-%d.%H:%M:%S')
    os.mkdir(os.getcwd() + "/" + timestampString)

    coordfile = open(timestampString + "/coords.xyz", "w")
    tempfile = open(timestampString + "/temperature.dat", "w")
    energyfile = open(timestampString + "/potential-energy.dat", "w")

    # headers
    tempfile.write("# time temperature\n")
    energyfile.write("# time potential-energy\n")

    time = 0.0  # initialise time

    # main MD loop
    for t in range(md.tsteps):
        print("#---- Time = ", round(time, 2), " ---- Steps = ", t, " ----")

        en = md.force()  # calculate forces
        md.integrate(t, en)  # integrate equations of motion

        # print current coordinates to file
        md.printcoords(time, coordfile)
        # print current temperature to file
        tempfile.write(str(round(time, 2)) + " " + str(md.temp) + "\n")
        # print current potential energy to file
        energyfile.write(str(round(time, 2)) + " " + str(md.energy_total) + "\n")

        time += md.dt  # increase time by dt

    md.statistics(tempfile, energyfile)  # calculate averages, SD, etc

    # close output files
    coordfile.close()
    tempfile.close()
    energyfile.close()


class MD(object):
    N = 100  # number of particles (integer for loop control)
    dN = float(N)  # number of particles (double for doing maths)
    boxLength = 25.0  # length of 1D box
    dim = 3.0  # dimensions

    """
    Initialise positions and velocities. 
    """
    def __init__(self):
        # declare the global variables

        # Constants
        self.lattice_step = self.boxLength / self.dN  # lattice spacing
        self.dtsteps = 500.0  # number of time steps (double for math)
        self.tsteps = int(self.dtsteps)  # number of time steps (integer for counting)
        self.dt = 0.01  # integration timestep
        self.rc = self.boxLength / 2  # distance cutoff for computing LJ interactions
        self.rc2 = self.rc ** 2  # distance cutoff squared
        self.ecut = 4.0 * ((self.rc ** -12) - (self.rc ** -6))  # value of LJ potential at r = rc: 4(1/rc^{12} - 1/rc^{6})

        # Values used to initialise the system
        starting_temp = 5.728  # temperature to scale lattice energies to initially
        initial_velocities = np.zeros([self.N, 3])  # Not updated as they are calculated fresh each dt

        # lists for storing stuff. All 3 dimensional for x, y, z
        self.coords = np.zeros([self.N, 3])  # coordinates
        self.prev_coords = np.zeros([self.N, 3])  # previous coordinates
        self.forces = np.zeros([self.N, 3])  # forces

        # data on the temperature and energy
        self.temp = starting_temp
        self.sum_temps = 0.0
        self.temps = np.zeros([self.tsteps])
        self.energy_total = 0.0
        self.sum_energy_totals = 0.0
        self.energy_totals = np.zeros([self.tsteps])

        print("#---- Initialising positions and velocities ----")

        sumv = np.zeros([3])  # sum of velocities
        sumv2 = 0.0  # sum of velocities squared

        # loop over the particles, giving each position and velocity
        for i in range(self.N):
            self.coords[i] = self.rand_lattice_pos()  # place particles on a lattice

            # assign velocities uniformly (but randomly) in range [-0.5, 0.5]
            initial_velocities[i] = np.random.rand(3) - 0.5

            sumv += initial_velocities[i]  # sum velocities
            sumv2 += np.sum(np.square(initial_velocities[i]))  # sum squared velocities

        sumv = sumv / self.dN  # finish calculating velocity of centre of mass
        sumv2 = sumv2 / self.dN  # mean-squared velocity (3 dims added together)

        # scale factor for velocities to achieve desired temperature
        sf = math.sqrt(self.dim * starting_temp / sumv2)

        for i in range(0, self.N):
            initial_velocities[i] = (initial_velocities[i] - sumv) * sf  # scale velocites
            self.prev_coords[i] = self.coords[i] - (initial_velocities[i] * self.dt)  # set previous positions

    # sets for initialising lattice positions
    x_positions = set(range(N))
    y_positions = set(range(N))
    z_positions = set(range(N))

    """
    Returns a random point in the lattice.
    Will not return the same point twice in a run. 
    """
    def rand_lattice_pos(self):
        # Take a random element from the set of possible positions
        x_pos = random.sample(self.x_positions, 1)[0]
        y_pos = random.sample(self.y_positions, 1)[0]
        z_pos = random.sample(self.z_positions, 1)[0]

        # Remove them from the sets so there's no repetition
        self.x_positions.remove(x_pos)
        self.y_positions.remove(y_pos)
        self.z_positions.remove(z_pos)

        pos = (np.array([x_pos, y_pos, z_pos]) + 0.5) * self.lattice_step
        return pos

    """
    Calculate forces on each particle using Lennard-Jones
    interactions.
    
    Returns the potential energy of the system
    """
    def force(self):
        # print("#---- Calculating forces ----")
        en = 0.0  # (re)set energy to zero

        # (re)set forces to zero
        self.forces = np.zeros([self.N, 3])

        # loop (inefficiently) over all pairs of atoms
        for i in range(0, self.N - 1):
            for j in range(i + 1, self.N):
                # get the closest image of coord j based on box conditions
                adjusted_j = self.find_closest_particle(i, j)

                r2 = self.dist_squared(self.coords[i], adjusted_j)  # square to compare to cutoff
                if r2 < self.rc2:  # test cutoff
                    # compute Lennard-Jones interaction from absolute distance
                    r2i = 1.0 / r2
                    r6i = r2i ** 3
                    ff = 48.0 * r2i * r6i * (r6i - 0.5)

                    # update forces in each dimension
                    self.forces[i] += ff * (self.coords[i] - adjusted_j)
                    self.forces[j] -= ff * (self.coords[i] - adjusted_j)

                    # update potential energy
                    en += 4.0 * r6i * (r6i - 1.0) - self.ecut
        return en

    """
    Finds the closest image of particle j to particle i.
    
    Checks the image 1 boxLength away in each axis, in each
    combination, returning the one with minimum distance from i
    """
    def find_closest_particle(self, i, j):
        coord_i = self.coords[i]
        coord_j = self.coords[j]

        # compare distance squared so we don't have to do expensive sqrt calcs
        min_dist2 = self.dist_squared(coord_i, coord_j)
        closest = coord_j
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    adjusted_j = coord_j + np.array([i, j, k]) * self.boxLength
                    dist2 = self.dist_squared(coord_i, adjusted_j)
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        closest = adjusted_j
        return closest

    @staticmethod
    def dist_squared(a, b):
        x_term = (a[0] - b[0]) ** 2
        y_term = (a[1] - b[1]) ** 2
        z_term = (a[2] - b[2]) ** 2
        return x_term + y_term + z_term

    """
    Integrate the equations of motion.
    
    Use Verlet algorithm to calculate new coordinates for each particle.
    Calculate and store the instantaneous temperature and potential
    energy of the system.
          
    """
    def integrate(self, t, en):
        # print("#---- Integrating equations of motion ----")

        sumv = np.zeros(3)
        sumv2 = 0.0

        for i in range(0, self.N):
            # Verlet algorithm to calculate new coordinates from forces and timestep
            new_coords = (2.0 * self.coords[i]) - self.prev_coords[i] + self.dt * self.dt * self.forces[i]

            # derive the velocity from it to calculate temperature and energy
            velocity = (new_coords - self.prev_coords[i]) / (2.0 * self.dt)
            sumv += velocity  # velocity centre of mass
            sumv2 += np.sum(np.square(velocity))  # total kinetic energy

            # Update coords for next timestep
            self.prev_coords[i] = self.coords[i]
            self.coords[i] = new_coords

        # calculate temp and store to calculate standard dev later
        self.temp = sumv2 / (self.dim * self.dN)  # instantaneous temperature
        self.sum_temps += self.temp
        self.temps[t] = self.temp

        # calculate energy and store to calculate standard dev later
        self.energy_total = (en + 0.5 * sumv2) / self.dN  # total energy per particle
        self.sum_energy_totals += self.energy_total
        self.energy_totals[t] = self.energy_total

    """
    Print coordinates to file
    """
    def printcoords(self, time, coordfile):
        coordfile.write('%d\n' % self.N)
        coordfile.write('time %10.10f\n' % time)
        for i in range(0, self.N):
            adjustedCoord = self.adjust_for_box(self.coords[i])
            coordfile.write('C %-8.8f %-8.8f %-8.8f\n' % (adjustedCoord[0], adjustedCoord[1], adjustedCoord[2]))

    """
    Shift a coordinate to be in the box between 0 and boxLength in each axis
    """
    def adjust_for_box(self, coord):
        adj_coord = np.zeros(3)

        for i in range(3):
            adj_value = coord[i] - self.boxLength * round(coord[i] / self.boxLength)
            if adj_value < 0:
                adj_value += self.boxLength

            adj_coord[i] = adj_value

        return adj_coord

    """
    Calculate average and standard deviation of temperature
    and potential energy, and print them to file.
    """
    def statistics(self, tfile, efile):
        # averages
        aveTemp = self.sum_temps / self.dtsteps
        aveEtot = self.sum_energy_totals / self.dtsteps

        # standard deviation
        varTemp = 0.0
        varEtot = 0.0
        for i in range(0, self.tsteps):
            varTemp += (self.temps[i] - aveTemp) ** 2
            varEtot += (self.energy_totals[i] - aveEtot) ** 2

        sdTemp = math.sqrt(varTemp / self.dtsteps)
        sdEtot = math.sqrt(varEtot / self.dtsteps)
        tfile.write('# Average temperature: %10.2f\n' % aveTemp)
        tfile.write('# Standard deviation: %10.2f\n' % sdTemp)
        efile.write('# Average total energy: %10.2f\n' % aveEtot)
        efile.write('# Standard deviation: %10.2f\n' % sdEtot)


md = MD()
main(md)
