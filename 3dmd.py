#!/usr/bin/python

###
# BIOINF703 Sem2 2019, Lab 2
# Sebastian Dunn, sdun067
#
# 3DMD: My attempt at making this a 3D simulation
#
#
#
#
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
    dN = float(N)  # number of particles (double for doing maths)
    boxLength = 50.0  # length of 1D box
    dim = 3.0  # dimensions

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
        self.energy_total = 0.0  # total energy (pot + kin)
        self.temp = 1.728  # temperature

        # lists for storing stuff
        self.coords = []  # coordinates
        self.prev_coords = []  # previous coordinates
        self.velocities = []  # velocities
        self.forces = []  # forces

        # store data for averaging
        self.sum_temps = 0.0
        self.sum_energy_totals = 0.0
        self.temps = []
        self.enery_totals = []

        # initialise the force list
        for i in range(self.N):
            self.forces.append(0.0)

        # initialise the lists to be of the correct size for data storage
        for i in range(self.tsteps):
            self.temps.append(0.0)
            self.enery_totals.append(0.0)
        print("#---- Initialising positions and velocities ----")

        sumv = velocities(0.0, 0.0, 0.0)  # sum of velocities
        sumv2 = 0.0  # sum of velocities squared

        # loop over the particles
        for i in range(self.N):
            # coords start in a very simple diagonal in 3D space. I want to focus on
            # getting the calculations working before a 3D random spacing algorithm
            position = self.lattice_pos(i)
            self.coords.append(coordinate(position, position, position))  # place particles on a lattice

            # assign velocities uniformly (but randomly) in range [-0.5, 0.5]
            self.velocities.append(velocities(self.random_v(), self.random_v(), self.random_v()))

            sumv = sumv.plus(self.velocities[i])  # sum velocities
            sumv2 += self.velocities[i].sumv2()  # sum squared velocities

        sumv = sumv.scale(1 / self.dN)  # finish calculating velocity of centre of mass
        sumv2 = sumv2 / self.dN  # mean-squared velocity (3 dims added together)
        sf = math.sqrt(self.dim * self.temp / sumv2)  # scale factor for velocites to achieve desired temperature

        for i in range(0, self.N):
            self.velocities[i] = self.velocities[i].minus(sumv).scale(sf)  # scale velocites
            self.prev_coords[i] = self.coords[i].move_by(self.velocities[i].scale(-1.0), self.dt) # set previous positions

    def random_v(self):
        return random.random() - 0.5

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
            self.forces[i] = 0.0

        # loop (inefficiently) over all pairs of atoms
        for i in range(0, self.N - 1):
            for j in range(i + 1, self.N):
                xr = self.coords[i].distance_from(self.coords[j])  # distance between atoms i and j

                xr -= self.boxLength * round(xr / self.boxLength)  # periodic boundary conditions

                r2 = xr ** 2  # square to compare to cutoff
                if r2 < self.rc2:  # test cutoff
                    # compute Lennard-Jones interaction
                    r2i = 1.0 / r2
                    r6i = r2i ** 3
                    ff = 48.0 * r2i * r6i * (r6i - 0.5)
                    # update forces
                    self.forces[i] += ff * xr
                    self.forces[j] -= ff * xr
                    # update potential energy
                    en += 4.0 * r6i * (r6i - 1.0) - self.ecut
        return en

    # integrate equations of motion
    def integrate(self, t, en):
        print("#---- Integrating equations of motion ----")
        sumv = 0.0
        sumv2 = 0.0
        for i in range(0, self.N): ##todo make forces a 3D object and do it properly here.
            new_coords = 2.0 * self.coords[i] - self.prev_coords[i] + self.dt * self.dt * self.forces[i]  # Verlet algorithm
            velocity = (new_coords - self.prev_coords[i]) / (2.0 * self.dt)  # velocity
            sumv += velocity  # velocity centre of mass
            sumv2 += velocity ** 2  # total kinetic energy
            self.prev_coords[i] = self.coords[i]  # update previous positions
            self.coords[i] = new_coords  # update current positions

        self.temp = sumv2 / (self.dim * self.dN)  # instantaneous temperature
        # store for calculating SD
        self.sum_temps += self.temp
        self.temps[t] = self.temp
        self.energy_total = (en + 0.5 * sumv2) / self.dN  # total energy per particle
        # store for calculating SD
        self.sum_energy_totals += self.energy_total
        self.enery_totals[t] = self.energy_total

    # print coordinates
    def printcoords(self, time, coordfile):
        coordfile.write('%d\n' % self.N)
        coordfile.write('time %10.10f\n' % time)
        for i in range(0, self.N):
            adjustedCoord = self.adjust_for_box(self.coords[i])
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
        aveTemp = self.sum_temps / self.dtsteps
        aveEtot = self.sum_energy_totals / self.dtsteps

        # standard deviation
        varTemp = 0.0
        varEtot = 0.0
        for i in range(0, self.tsteps):
            varTemp += (self.temps[i] - aveTemp) ** 2
            varEtot += (self.enery_totals[i] - aveEtot) ** 2

        sdTemp = math.sqrt(varTemp / self.dtsteps)
        sdEtot = math.sqrt(varEtot / self.dtsteps)
        tfile.write('# Average temperature: %10.2f\n' % aveTemp)
        tfile.write('# Standard deviation: %10.2f\n' % sdTemp)
        efile.write('# Average total energy: %10.2f\n' % aveEtot)
        efile.write('# Standard deviation: %10.2f\n' % sdEtot)


class coordinate:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def plus(self, them):
        return coordinate(self.x + them.x, self.y + them.y, self.z + them.z)

    def move_by(self, velocities, dt):
        return coordinate(self.x + velocities.xv * dt,
                          self.y + velocities.yv * dt,
                          self.z + velocities.zv * dt)

    def scale(self, sf):
        return coordinate(self.x * sf, self.y * sf, self.z * sf)

    def distance_from(self, them):
        x_term = (self.x - them.x) ** 2
        y_term = (self.y - them.y) ** 2
        z_term = (self.z - them.z) ** 2
        return math.sqrt(x_term + y_term + z_term)


class velocities:
    def __init__(self, xv, yv, zv):
        self.xv = xv
        self.yv = yv
        self.zv = zv

    def plus(self, them):
        return velocities(self.xv + them.xv, self.yv + them.yv, self.zv + them.zv)

    def minus(self, them):
        return velocities(self.xv - them.xv, self.yv - them.yv, self.zv - them.zv)

    def scale(self, sf):
        return velocities(self.xv * sf, self.yv * sf, self.zv * sf)

    def sumv2(self):
        return self.xv ** 2 + self.yv ** 2 + self.zv ** 2

class forces:
    def __init__(self, xf, yf, zf):
        self.xf = xf
        self.yf = yf
        self.zf = zf


md = MD()
main(md)
