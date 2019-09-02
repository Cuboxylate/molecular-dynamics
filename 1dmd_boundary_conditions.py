#!/usr/bin/python

"""
BIOINF703 Sem2 2019, Lab 2
Sebastian Dunn, sdun067

1DMD: Corrected periodic boundary conditions

This version of the script corrects a bug
we found in the lab:

Bug:
    Atoms pass through box boundaries during
    simulation.

Fix:
    Adjust atom coordinates to be between 0
    and 1 box length when printing to file.
    (see printcoords() and adjust_for_box()
    from line 234)

See an example run showing this in
boundary_conditions_output

Other changes:
    Each run will be output to a separate timestamped
    subfolder now
    Tidied up - added comments and changed variable
    names to make it easier for first time eyes

"""

# import needed modules
import random
import math
from datetime import datetime
import os

"""
The main loop function
"""
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
        energyfile.write(str(round(time, 2)) + " " + str(md.etot) + "\n")

        time += md.dt  # increase time by dt

    md.statistics(tempfile, energyfile)  # calculate averages, SD, etc

    # close output files
    coordfile.close()
    tempfile.close()
    energyfile.close()


class MD(object):
    N = 15  # number of particles (integer for loop control)
    dN = float(N)  # number of particles (double for doing maths)
    boxLength = 50.0  # length of 1D box
    dim = 1.0  # dimensions

    # initialise positions and velocites
    def __init__(self):
        # declare the global variables

        # constants
        self.lattice_step = self.boxLength / self.dN  # lattice spacing
        self.dtsteps = 750.0  # number of time steps (double for math)
        self.tsteps = int(self.dtsteps)  # number of time steps (integer for counting)
        self.dt = 0.01  # integration timestep
        self.rc = self.boxLength / 2  # distance cutoff for computing LJ interactions
        self.rc2 = self.rc ** 2  # distance cutoff squared
        self.ecut = 4.0 * (
                    (self.rc ** -12) - (self.rc ** -6))  # value of LJ potential at r = rc: 4(1/rc^{12} - 1/rc^{6})

        # Values used to initialise the system
        starting_temp = 5.728  # initial temperature
        initial_v = []  # initial velocities

        # lists for storing stuff
        self.x_coords = []  # coordinates
        self.x_prev = []  # previous coordinates
        self.forces = []  # forces

        # data on the temperature and energy
        self.temp = starting_temp
        self.sum_temp = 0.0
        self.temps = []
        self.energy_tot = 0.0
        self.sum_energy_tot = 0.0
        self.energy_tots = []

        # initialise the lists to be the correct size
        for i in range(self.N):
            initial_v.append(0.0)
            self.x_coords.append(0.0)
            self.x_prev.append(0.0)
            self.forces.append(0.0)

        # initialise the lists to be of the correct size for data storage
        for i in range(self.tsteps):
            self.temps.append(0.0)
            self.energy_tots.append(0.0)

        print("#---- Initialising positions and velocities ----")

        sumv = 0.0  # sum of velocities
        sumv2 = 0.0  # sum of velocities squared

        # loop over the particles
        for i in range(self.N):
            self.x_coords[i] = self.lattice_pos(i)  # place particles on a lattice
            initial_v[i] = random.random() - 0.5  # assign velocities uniformly (but randomly) in range [-0.5, 0.5]
            sumv += initial_v[i]  # sum velocities
            sumv2 += initial_v[i] ** 2  # sum squared velocities

        sumv = sumv / self.dN  # velocity of centre of mass
        sumv2 = sumv2 / self.dN  # mean-squared velocity

        # scale factor for velocities to achieve desired temperature
        sf = math.sqrt(self.dim * starting_temp / sumv2)

        for i in range(0, self.N):
            # scale initial velocities. These aren't actually used again as new coords are computed from forces alone
            initial_v[i] = (initial_v[i] - sumv) * sf
            self.x_prev[i] = self.x_coords[i] - initial_v[i] * self.dt  # set previous positions

    """
    Place particles on a lattice
    """
    def lattice_pos(self, i):
        pos = (i + 0.5) * self.lattice_step
        return pos

    """
    Calculate forces on each particle using Lennard-Jones
    interactions.
    
    Returns the potential energy of the system
    """
    def force(self):
        print("#---- Calculating forces ----")
        en = 0.0  # set energy to zero

        # (re)set forces to zero
        for i in range(self.N):
            self.forces[i] = 0.0

        # loop (inefficiently) over all pairs of atoms
        for i in range(0, self.N - 1):
            for j in range(i + 1, self.N):
                dist = self.x_coords[i] - self.x_coords[j]  # distance between atoms i and j

                dist -= self.boxLength * round(dist / self.boxLength)  # periodic boundary conditions

                r2 = dist ** 2  # square to compare to cutoff
                if r2 < self.rc2:  # test cutoff
                    # compute Lennard-Jones interaction
                    r2i = 1.0 / r2
                    r6i = r2i ** 3
                    ff = 48.0 * r2i * r6i * (r6i - 0.5)

                    # update forces
                    self.forces[i] += ff * dist
                    self.forces[j] -= ff * dist

                    # update potential energy
                    en += 4.0 * r6i * (r6i - 1.0) - self.ecut
        return en

    """
    Integrate the equations of motion.
    
    Use Verlet algorithm to calculate new coordinates for each particle.
    Calculate and store the instantaneous temperature and potential
    energy of the system.
          
    """
    def integrate(self, t, en):
        print("#---- Integrating equations of motion ----")

        sumv = 0.0
        sumv2 = 0.0

        for i in range(0, self.N):
            # Verlet algorithm to calculate new coordinates from forces and timestep
            x_new = 2.0 * self.x_coords[i] - self.x_prev[i] + self.dt * self.dt * self.forces[i]

            # derive the velocity from it to calculate temperature and energy
            velocity = (x_new - self.x_prev[i]) / (2.0 * self.dt)
            sumv += velocity  # velocity centre of mass
            sumv2 += velocity ** 2  # total kinetic energy

            # Update coords for next timestep
            self.x_prev[i] = self.x_coords[i]
            self.x_coords[i] = x_new

        # calculate temp and store to calculate standard dev later
        self.temp = sumv2 / (self.dim * self.dN)  # instantaneous temperature
        self.sum_temp += self.temp
        self.temps[t] = self.temp

        # calculate energy and store to calculate standard dev later
        self.energy_tot = (en + 0.5 * sumv2) / self.dN  # total energy per particle
        self.sum_energy_tot += self.energy_tot
        self.energy_tots[t] = self.energy_tot

    """
    Print coordinates to file
    """
    def printcoords(self, time, coordfile):
        coordfile.write('%d\n' % self.N)
        coordfile.write('time %10.10f\n' % time)
        for i in range(0, self.N):
            adjustedCoord = self.adjust_for_box(self.x_coords[i])
            coordfile.write('C %-8.8f 0.0 0.0\n' % adjustedCoord)

    """
    Shift a coordinate to be in the box between 0 and boxLength
    """
    def adjust_for_box(self, coord):
        adj_coord = coord - self.boxLength * round(coord / self.boxLength)
        if (adj_coord < 0):
            adj_coord += self.boxLength

        return adj_coord

    """
    Calculate average and standard deviation of temperature
    and potential energy, and print them to file.
    """
    def statistics(self, tfile, efile):
        # averages
        aveTemp = self.sum_temp / self.dtsteps
        aveEtot = self.sum_energy_tot / self.dtsteps

        # standard deviation
        varTemp = 0.0
        varEtot = 0.0
        for i in range(0, self.tsteps):
            varTemp += (self.temps[i] - aveTemp) ** 2
            varEtot += (self.energy_tots[i] - aveEtot) ** 2

        sdTemp = math.sqrt(varTemp / self.dtsteps)
        sdEtot = math.sqrt(varEtot / self.dtsteps)
        tfile.write('# Average temperature: %10.2f\n' % aveTemp)
        tfile.write('# Standard deviation: %10.2f\n' % sdTemp)
        efile.write('# Average total energy: %10.2f\n' % aveEtot)
        efile.write('# Standard deviation: %10.2f\n' % sdEtot)


md = MD()
main(md)
