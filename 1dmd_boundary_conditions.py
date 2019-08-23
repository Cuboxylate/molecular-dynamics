#!/usr/bin/python

###
# BIOINF703 Sem2 2019, Lab 2
# Sebastian Dunn, sdun067
#
# 1DMD: Corrected periodic boundary conditions
#
# This version of the script corrects a bug
# we found in the lab:
#
# Bug:
#   Atoms pass through box boundaries during
#   simulation.
#
# Fix:
#   Adjust atom coordinates to be between 0
#   and 1 box length when printing to file.
#   (see printcoords() and adjust_for_box()
#   from line 196)
#
# See an example run showing this in
# boundary_conditions_output
#
# Also note that this version will make outputs
# in timestamped subfolders, so we're not
# overwriting our files each run. It assumes
# write permission on the working directory.
###

# import needed modules
import random
import math
from datetime import datetime
import os

# the main loop function
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
    dN = float(N) # number of particles (double for doing maths)
    boxLength = 50.0  # length of 1D box
    dim = 1.0  # dimensions

    # initialise positions and velocites
    def __init__(self):

        # declare the global variables
        # constants
        self.a = self.boxLength / self.dN  # lattice spacing
        self.dtsteps = 750.0  # number of time steps (double for math)
        self.tsteps = int(self.dtsteps)  # number of time steps (integer for counting)
        self.dt = 0.01  # integration timestep
        self.rc = self.boxLength / 2  # distance cutoff for computing LJ interactions
        self.rc2 = self.rc ** 2  # distance cutoff squared
        self.ecut = 4.0 * ((self.rc ** -12) - (self.rc ** -6))  # value of LJ potential at r = rc: 4(1/rc^{12} - 1/rc^{6})

        # values that change during the simulation
        self.en = 0.0  # potential energy
        self.etot = 0.0  # total energy (pot + kin)
        self.temp = 5.728  # temperature

        # lists for storing stuff
        self.x = []  # coordinates
        self.xp = []  # previous coordinates
        self.v = []  # velocities
        self.f = []  # forces

        # store data for averaging
        self.sumTemp = 0.0
        self.sumEtot = 0.0
        self.temps = []
        self.etots = []

        # initialise the lists to be the correct size
        for i in range(self.N):
            self.x.append(0.0)
            self.xp.append(0.0)
            self.v.append(0.0)
            self.f.append(0.0)

        # initialise the lists to be of the correct size for data storage
        for i in range(self.tsteps):
            self.temps.append(0.0)
            self.etots.append(0.0)
        print("#---- Initialising positions and velocities ----")

        sumv = 0.0  # sum of velocities
        sumv2 = 0.0  # sum of velocities squared

        # loop over the particles
        for i in range(self.N):
            self.x[i] = self.lattice_pos(i)  # place particles on a lattice
            self.v[i] = random.random() - 0.5  # assign velocities uniformly (but randomly) in range [-0.5, 0.5]
            sumv += self.v[i]  # sum velocities
            sumv2 += self.v[i] ** 2  # sum squard velocities

        sumv = sumv / self.dN  # velocity of centre of mass
        sumv2 = sumv2 / self.dN  # mean-squared velocity
        sf = math.sqrt(self.dim * self.temp / sumv2)  # scale factor for velocites to achieve desired temperature

        for i in range(0, self.N):
            self.v[i] = (self.v[i] - sumv) * sf  # scale velocites
            self.xp[i] = self.x[i] - self.v[i] * self.dt  # set previous positions

    # place particles on a lattice
    def lattice_pos(self, i):
        pos = (i + 0.5) * self.a
        return pos

    # calculate forces
    def force(self):
        print("#---- Calculating forces ----")
        en = 0.0  # (re)set energy to zero
        # (re)set forces to zero
        for i in range(self.N):
            self.f[i] = 0.0

        # loop (inefficiently) over all pairs of atoms
        for i in range(0, self.N - 1):
            for j in range(i + 1, self.N):
                xr = self.x[i] - self.x[j]  # distance between atoms i and j

                xr -= self.boxLength * round(xr / self.boxLength)  # periodic boundary conditions

                r2 = xr ** 2  # square to compare to cutoff
                if r2 < self.rc2:  # test cutoff
                    # compute Lennard-Jones interaction
                    r2i = 1.0 / r2
                    r6i = r2i ** 3
                    ff = 48.0 * r2i * r6i * (r6i - 0.5)
                    # update forces
                    self.f[i] += ff * xr
                    self.f[j] -= ff * xr
                    # update potential energy
                    en += 4.0 * r6i * (r6i - 1.0) - self.ecut
        return en

    # integrate equations of motion
    def integrate(self, t, en):
        print("#---- Integrating equations of motion ----")
        sumv = 0.0
        sumv2 = 0.0
        for i in range(0, self.N):
            xx = 2.0 * self.x[i] - self.xp[i] + self.dt * self.dt * self.f[i]  # Verlet algorithm
            vi = (xx - self.xp[i]) / (2.0 * self.dt)  # velocity
            sumv += vi  # velocity centre of mass
            sumv2 += vi ** 2  # total kinetic energy
            self.xp[i] = self.x[i]  # update previous positions
            self.x[i] = xx  # update current positions

        self.temp = sumv2 / (self.dim * self.dN)  # instantaneous temperature
        # store for calculating SD
        self.sumTemp += self.temp
        self.temps[t] = self.temp
        self.etot = (en + 0.5 * sumv2) / self.dN  # total energy cper particle
        # store for calculating SD
        self.sumEtot += self.etot
        self.etots[t] = self.etot

    # print coordinates
    def printcoords(self, time, coordfile):
        coordfile.write('%d\n' % self.N)
        coordfile.write('time %10.10f\n' % time)
        for i in range(0, self.N):
            adjustedCoord = self.adjust_for_box(self.x[i])
            coordfile.write('C %-8.8f 0.0 0.0\n' % adjustedCoord)

    # shift a coordinate to be between 0 and boxLength
    def adjust_for_box(self, coord):
        adj_coord = coord - self.boxLength * round(coord / self.boxLength)
        if (adj_coord < 0):
            adj_coord += self.boxLength

        return adj_coord

    # calculate averages, etc and print to file
    def statistics(self, tfile, efile):
        # averages
        aveTemp = self.sumTemp / self.dtsteps
        aveEtot = self.sumEtot / self.dtsteps

        # standard deviation
        varTemp = 0.0
        varEtot = 0.0
        for i in range(0, self.tsteps):
            varTemp += (self.temps[i] - aveTemp) ** 2
            varEtot += (self.etots[i] - aveEtot) ** 2

        sdTemp = math.sqrt(varTemp / self.dtsteps)
        sdEtot = math.sqrt(varEtot / self.dtsteps)
        tfile.write('# Average temperature: %10.2f\n' % aveTemp)
        tfile.write('# Standard deviation: %10.2f\n' % sdTemp)
        efile.write('# Average total energy: %10.2f\n' % aveEtot)
        efile.write('# Standard deviation: %10.2f\n' % sdEtot)


md = MD()
main(md)
