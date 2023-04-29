#!/usr/bin/env python
"""orient.py: A simple commandline xyz manipulation tool."""

#
#  Orient molecule: simple commandline xyz manipulation
#  Copyright (C) 2013-2023  Shane M. Parker <shane.parker@case.edu>

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys

import math as m
import numpy as np

bohr2ang = 0.529177210903

DEBUG = False

# masses of most common isotopes to 3 decimal points, from
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
# yapf: disable
masses = {
    'x' : 0.000,
    'h' : 1.008,
    'he': 4.003,
    'li': 7.016,
    'be': 9.012,
    'b' : 11.009,
    'c' : 12.000,
    'n' : 14.003,
    'o' : 15.995,
    'f' : 18.998,
    'ne': 19.992,
    'na': 22.990,
    'mg': 23.985,
    'al': 26.981,
    'si': 27.977,
    'p' : 30.974,
    's' : 31.972,
    'cl': 34.969,
    'ar': 39.962,
    'k' : 38.964,
    'ca': 39.963,
    'sc': 44.956,
    'ti': 47.948,
    'v' : 50.944,
    'cr': 51.941,
    'mn': 54.938,
    'fe': 55.935,
    'co': 58.933,
    'ni': 57.935,
    'cu': 62.930,
    'zn': 63.929,
    'ga': 68.926,
    'ge': 73.921,
    'as': 74.922,
    'se': 79.917,
    'br': 78.918,
    'kr': 83.912,
    'rb': 84.912,
    'sr': 87.906,
    'y': 88.906,
    'zr': 89.905,
    'nb': 92.906,
    'mo': 97.905,
    'tc': 98.906,
    'ru': 101.904,
    'rh': 102.906,
    'pd': 107.904,
    'ag': 106.905,
    'cd': 113.903,
    'in': 114.904,
    'sn': 119.902,
    'sb': 120.904,
    'te': 129.906,
    'i' : 126.904,
    'xe': 131.904,
    'cs': 132.905,
    'ba': 137.905,
    'la': 138.906,
    'ce': 139.905,
    'pr': 140.908,
    'nd': 141.907,
    'pm': 145.914,
    'sm': 151.920,
    'eu': 152.921,
    'gd': 157.924,
    'tb': 158.925,
    'dy': 163.929,
    'ho': 164.930,
    'er': 165.930,
    'tm': 168.934,
    'yb': 173.939,
    'lu': 174.941,
    'hf': 179.947,
    'ta': 180.948,
    'w' : 183.951,
    're': 186.956,
    'os': 191.961,
    'ir': 192.963,
    'pt': 194.965,
    'au': 196.967,
    'hg': 201.971,
    'tl': 204.974,
    'pb': 207.976,
    'bi': 208.980,
    'po': 209.483,
    'at': 210.487,
    'rn': 217.673,
    'fr': 223.020,
    'ra': 225.274,
    'ac': 227.028,
    'th': 231.036,
    'pa': 231.036,
    'u' : 238.051,
    'np': 236.547,
    'pu': 240.723,
    'am': 242.059,
    'cm': 245.567,
    'bk': 248.073,
    'cf': 250.578,
    'es': 252.083,
    'fm': 257.095,
    'md': 259.101,
    'no': 259.101,
    'lr': 262.110,
    'rf': 267.122,
    'db': 268.126,
    'sg': 271.134,
    'bh': 272.138,
    'hs': 270.134,
    'mt': 276.152,
    'ds': 281.165,
    'rg': 280.165,
    'cn': 285.177,
    'nh': 284.179,
    'fl': 289.190,
    'mc': 288.193,
    'lv': 293.204,
    'ts': 292.207,
    'og': 294.214
}
# yapf: enable

# list of all elements, sorted by atomic number
elements = [
    'x', 'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca',
    'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',
    'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba', 'la', 'ce',
    'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir',
    'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm',
    'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'nh', 'fl', 'mc',
    'lv', 'ts', 'og'
]


class Geometry_:
    '''Stores all of the data in an xyz file'''

    def __init__(self):
        self.coordinates = None
        self.natoms = None
        self.comment = None
        self.mass = None

        self.origin = None
        self.axes = None

    def compute_center_of_mass(self):
        '''
        Returns the center of mass of the geometry.
        '''
        com = np.dot(self.mass, self.coordinates) / np.sum(self.mass)
        return com

    def compute_inertia(self):
        '''Returns the moment of inertia tensor'''
        com = self.compute_center_of_mass()
        data = self.coordinates - com

        inertial_tensor = np.einsum("ax,a,ay->xy", data, self.mass, data)
        inertial_tensor *= -1
        return inertial_tensor

    def apply(self, operation):
        """Apply a translation/rotation/reflection to the geometry"""
        raise NotImplementedError


class GeometryXYZ(Geometry_):
    """XYZ files"""

    def __init__(self, names, coordinates, comment="", extras=None):
        super().__init__()

        self.names = names
        self.coordinates = coordinates
        self.natoms = coordinates.shape[0]
        self.comment = comment
        self.extras = extras
        self.mass = [masses[n.lower()] for n in names]

        self.origin = None
        self.axes = None

    def apply(self, operation):
        """Apply a translation/rotation/reflection to the geometry"""
        operation(self.coordinates)

    def print(self):
        """print in xyz format"""
        print(f"{self.natoms:d}")
        print(self.comment)

        for i in range(self.natoms):
            extra = " ".join([f"{x:16.10f}" for x in self.extras[i]]) if self.extras else ""
            x, y, z = self.coordinates[i, :]
            print(f"{self.names[i]:>3s} {x:16.10f} {y:16.10f} {z:16.10f} {extra:s}")


class GeometryCube(Geometry_):
    """Cube file"""

    def __init__(self,
                 atomnumber,
                 charges,
                 coordinates,
                 origin,
                 nval,
                 resolutions,
                 axes,
                 volume_data,
                 comments=["", ""],
                 expect_dset=False):
        super().__init__()

        self.atomnumber = atomnumber
        self.charges = charges
        self.coordinates = np.array(coordinates)
        self.natoms = coordinates.shape[0]
        self.origin = np.array(origin).reshape([1,3])
        self.nval = nval
        self.resolutions = resolutions
        self.axes = np.array(axes)
        self.volume_data = volume_data
        self.comments = comments
        self.expect_dset = expect_dset

        self.mass = [masses[elements[i]] for i in atomnumber]

    def apply(self, operation):
        """Apply a translation/rotation/reflection to the geometry"""
        static_op = operation(self.coordinates)
        static_op(self.origin)

        # axes should not get translated, but every other action should happen
        if isinstance(operation, Rotate):
            static_op(self.axes)
        elif isinstance(operation, Reflect):
            static_op(self.axes)
        elif isinstance(operation, ShiftedOperation):
            # extract the non-translation part of the operation:w
            static_op.operation(self.axes)

    def print(self):
        """print in cube format"""
        # 1st and 2nd lines are comments
        for c in self.comments:
            print(c)

        # 3rd line is <natoms> <origin x> <origin y> <origin z>
        natoms = self.natoms
        if self.expect_dset:
            natoms *= -1
        origin = self.origin.reshape(3) / bohr2ang # origin should always be in bohr
        nvalstr = f" {self.nval:s}" if self.nval else ""
        print(f"{natoms:>5d} {origin[0]:>11.6f} {origin[1]:>11.6f} {origin[2]:>11.6f}{nvalstr:>5s}".rstrip())

        # 4th, 5th, 6th are <n1> <v1> <v2> <v3>
        for i in range(3):
            res = self.resolutions[i]
            axis = self.axes[i, :] # unit on axes shouldn't matter
            print(f"{res:>5s} {axis[0]:>11.6f} {axis[1]:>11.6f} {axis[2]:>11.6f}")

        # next natoms lines define the molecule as
        # <atom number> <charge> <x> <y> <z>
        for i in range(self.natoms):
            atom = self.atomnumber[i]
            xyz = self.coordinates[i, :] / bohr2ang # coordinates always in bohr
            print(f"{atom:>5d} {self.charges[i]:>11s} {xyz[0]:>11.6f} {xyz[1]:>11.6f} {xyz[2]:>11.6f}")

        for v in self.volume_data:
            print(v)


def read_xyz(filename):
    '''reads xyz file and returns GeometryXYZ object'''
    out = []

    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            natoms = int(line)
            comment = f.readline().rstrip()

            names = []
            coords = []
            extras = []

            for i in range(natoms):
                line = f.readline()
                data = line.split()
                name, x, y, z = data[0:4]
                extra = [float(d) for d in data[4:]]

                names.append(name.capitalize())
                coords.append([float(x), float(y), float(z)])
                if extra:
                    extras.append(extra)

            out.append(GeometryXYZ(names, np.array(coords), comment=comment, extras=extras))

            line = f.readline()

    return out


def read_cube(filename):
    """reads cube format and returns GeometryCube"""
    out = []

    with open(filename, "r") as f:
        # 1st and 2nd lines are comments
        comments = [f.readline().rstrip(), f.readline().rstrip()]

        # 3rd line is <natoms> <origin x> <origin y> <origin z>
        line = f.readline()
        split = line.split()
        natoms, *origin = split[0:4]
        nval = split[4] if len(split) > 4 else None
        natoms = int(natoms)
        expect_dset = natoms < 0
        natoms = abs(natoms)
        origin = np.array([float(x) for x in origin]) * bohr2ang # cube has bohr, want ang

        # 4th, 5th, 6th are <n1> <v1> <v2> <v3>
        resolutions = []
        axes = np.zeros([3, 3])
        for i in range(3):
            res, *axis = f.readline().split()
            resolutions.append(res) # sign on resolutions determines unit, but units don't matter here
            axes[i, :] = np.array([float(x) for x in axis])

        coords = np.zeros([natoms, 3])
        atomnumber = []
        charges = []
        # next natoms lines define the molecule as
        # <atom number> <charge> <x> <y> <z>
        for i in range(natoms):
            atom, chg, *xyz = f.readline().split()
            atomnumber.append(int(atom))
            charges.append(chg)

            xyz = np.array([float(x) for x in xyz]) * bohr2ang # cube has bohr, want ang
            coords[i, :] = xyz

        volume_lines = []
        # the rest is the volume data
        while True:
            line = f.readline()
            if line == "":
                break
            volume_lines.append(line.rstrip())

        cube = GeometryCube(atomnumber,
                            charges,
                            coords,
                            origin,
                            nval,
                            resolutions,
                            axes,
                            volume_lines,
                            comments,
                            expect_dset=expect_dset)
        out.append(cube)

    return out


def read_file(filename):
    if filename.endswith(".xyz"):
        return read_xyz(filename)
    elif filename.endswith(".cub") or filename.endswith(".cube"):
        return read_cube(filename)


class Operation:
    '''Base class for generic operation'''

    def __call__(self, data):
        '''act on provided coordinate data'''
        raise Exception("Improper use of Operation class!")

    def iscomposable(self, op):
        '''do not compose any objects by default'''
        return False

    def compose(self, op):
        '''compose this op and input op'''
        raise RuntimeError("This operation not composable")


#-------------------------------------------------------------------------------------------#
# Translation classes                                                                       #
#-------------------------------------------------------------------------------------------#


class Translate(Operation):
    """Base class for translation operations"""

    def __call__(self, data):
        displacement = self.displacement_func(data)
        data += np.repeat(displacement.reshape(1, 3), data.shape[0], axis=0)
        return StaticTranslate(displacement)

    def displacement_func(self, data):
        """dummy displacement function"""
        raise NotImplementedError


class StaticTranslate(Translate):
    """Translate with fully specified information"""

    def __init__(self, displacement):
        self.displacement = displacement

    def displacement_func(self, data):
        return self.displacement

    def iscomposable(self, op):
        return isinstance(op, StaticTranslate)

    def compose(self, trans):
        assert self.iscomposable(trans)

        self.displacement += trans.displacement


class AtomTranslate(Translate):
    """Translate so specified atom is at the origin"""

    def __init__(self, iatom, fac=1.0):
        self.iatom = iatom
        self.fac = fac

    def displacement_func(self, data):
        return self.fac * data[self.iatom, :]


class CentroidTranslate(Translate):
    """Translate so the weighted average of atoms is at the origin"""

    def __init__(self, atomlist, fac=1.0):
        self.atomlist = atomlist
        self.fac = fac / len(atomlist)

    def displacement_func(self, data):
        return np.sum(data[self.atomlist, :], axis=0) * self.fac


class COMTranslate(Translate):
    """Translate so molecule center of mass is at the origin"""

    def __init__(self, geom):
        self.mass = geom.mass

    def displacement_func(self, data):
        return -np.dot(self.mass, data) / np.sum(self.mass)


#-------------------------------------------------------------------------------------------#
# Rotation classes                                                                          #
#-------------------------------------------------------------------------------------------#
class Rotate(Operation):
    '''Generic rotation'''

    def __call__(self, data):
        A = self.rotate_func(data)
        detA = np.linalg.det(A)
        if detA < 0.0:
            raise Exception("Determinant of Rotation needs to be 1")
        tmp = np.dot(data, A.T)
        data[:] = tmp[:]
        return StaticRotate(A)

    def rotate_func(self, data):
        """dummy rotate function"""
        raise NotImplementedError

    @staticmethod
    def axis_angle(axis, angle):
        axis /= np.linalg.norm(axis)
        theta = m.radians(angle)

        costheta = m.cos(theta)
        sintheta = m.sin(theta)

        Ex = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.0]])

        A = costheta * \
            np.eye(3) + (1.0 - costheta) * \
            np.dot(axis.reshape(3, 1), axis.reshape(1, 3)) + \
            sintheta * Ex
        return A


class StaticRotate(Rotate):
    '''Rotate based on static information'''

    def __init__(self, rot_matrix):
        self.rot_matrix = rot_matrix

    def rotate_func(self, data):
        return self.rot_matrix

    def iscomposable(self, op):
        return isinstance(op, StaticRotate)

    def compose(self, rot):
        assert self.iscomposable(rot)

        self.rot_matrix = np.dot(self.rot_matrix, rot.rot_matrix)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """make rotation matrix from axis and angle"""
        return cls(Rotate.axis_angle(axis, angle))


class AtomPairRotate(Rotate):
    """Rotate around axis defined by vector between two atoms"""

    def __init__(self, i, j, angle):
        self.i, self.j = i, j
        self.angle = angle

    def rotate_func(self, data):
        """compute rotation matrix from input data"""
        iatom = data[self.i, :]
        jatom = data[self.j, :]
        axis = jatom - iatom

        return Rotate.axis_angle(axis, self.angle)


class AlignRotate(Rotate):
    """Rotate so that the vector between two atoms is aligned along xyz"""

    def __init__(self, i, j, k):
        self.i, self.j, self.k = i, j, k

    def rotate_func(self, data):
        """compute rotation matrix from input data"""
        iatom = data[self.i, :]
        jatom = data[self.j, :]
        katom = data[self.k, :]

        vec1 = jatom[:] - iatom[:]
        vec1 /= np.linalg.norm(vec1)

        vec2 = katom[:] - iatom[:]
        vec2 -= np.dot(vec1, vec2) * vec1[:]
        vec2 /= np.linalg.norm(vec2)

        vec3 = np.cross(vec1, vec2)

        return np.array([vec1[:], vec2[:], vec3[:]])


class InertiaRotate(Rotate):
    """Rotate so that the moments of inertia are aligned along xyz"""

    def __init__(self, geom):
        self.mass = geom.mass

    def rotate_func(self, data):
        """compute rotation matrix from input data"""
        inertial_tensor = np.einsum("ax,a,ay->xy", data, self.mass, data)
        inertial_tensor *= -1
        # negate sign to reverse the sorting of the tensor
        eig, axes = np.linalg.eigh(-inertial_tensor)
        axes = axes.T

        # adjust sign of axes so third moment moment is positive new in X, and Y axes
        testcoords = np.dot(data, axes.T)  # a little wasteful, but fine for now
        thirdmoment = np.einsum("ax,a->x", testcoords**3, self.mass)

        for i in range(2):
            if thirdmoment[i] < 1.0e-6:
                axes[i, :] *= -1.0

        # rotation matrix must have determinant of 1
        if np.linalg.det(axes) < 0.0:
            axes[2, :] *= -1.0

        return axes


class NormalRotate(Rotate):
    """Rotate about plane defined by a list of atoms"""

    def __init__(self, atomlist, angle):
        self.atomlist = atomlist
        self.angle = angle

    def rotate_func(self, data):
        """compute rotation matrix"""
        atoms = np.array([data[i, :] for i in self.atomlist])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2, :]
        normal /= np.linalg.norm(normal)

        # compute plane of first three atoms to specify the direction
        v1 = data[self.atomlist[1], :] - data[self.atomlist[0], :]
        v2 = data[self.atomlist[2], :] - data[self.atomlist[0], :]
        crossnormal = np.cross(v1, v2)

        if np.dot(normal, crossnormal) < 0.0:
            normal *= -1.0

        return Rotate.axis_angle(normal, self.angle)


class PlaneRotate(Rotate):
    """Rotate about plane defined by a triple of atoms"""

    def __init__(self, atomlist):
        self.atomlist = atomlist

    def rotate_func(self, data):
        """compute rotation matrix from input data"""
        atoms = np.array([data[i, :] for i in self.atomlist])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2, :]
        normal /= np.linalg.norm(normal)

        # compute plane of first three atoms to specify the direction
        v1 = data[self.atomlist[1], :] - data[self.atomlist[0], :]
        v2 = data[self.atomlist[2], :] - data[self.atomlist[0], :]
        crossnormal = np.cross(v1, v2)

        if np.dot(normal, crossnormal) < 0.0:
            normal *= -1.0

        xvec = atoms[1, :] - atoms[0, :]
        vec1 = xvec - np.dot(xvec, normal) * normal
        vec1 /= np.linalg.norm(vec1)

        vec2 = np.cross(normal, vec1)

        A = np.array([vec1, vec2, normal])
        if np.linalg.det(A) < 0.0:
            A[2, :] *= -1.0
        return A


#-------------------------------------------------------------------------------------------#
# Reflection classes                                                                        #
#-------------------------------------------------------------------------------------------#


class Reflect(Operation):
    '''Reflect across a plane'''

    def __call__(self, data):
        '''Reflect by decomposing position as position on plane plus
        vector parallel to normal. Then, change sign on vector
        parallel to normal'''
        normal = self.reflect_func(data)
        normal /= np.linalg.norm(normal)

        # implemented like a householder reflection
        proj = np.dot(data, normal)
        data -= 2.0 * np.dot(proj.reshape(len(proj), 1), normal.reshape(1, 3))

        return StaticReflect(normal)

    def reflect_func(self, data):
        """dummy reflect function"""
        raise NotImplementedError


class StaticReflect(Reflect):
    '''Reflect across a statically defined plane'''

    def __init__(self, normal):
        # force the normal to have unit norm
        self.normal = normal / np.linalg.norm(normal)

    def reflect_func(self, data):
        return self.normal


class BondReflect(Reflect):
    '''Reflect across normal defined by bond'''

    def __init__(self, iatom, jatom):
        self.iatom = iatom
        self.jatom = jatom

    def reflect_func(self, data):
        return data[self.jatom, :] - data[self.iatom, :]


class PlaneReflect(Reflect):
    '''Reflect across a plane fitted to a group of atoms
    Undefine results if used away from origin'''

    def __init__(self, atomlist):
        self.atomlist = atomlist

    def reflect_func(self, data):
        atoms = np.array([data[i, :] for i in self.atomlist])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2, :]
        normal /= np.linalg.norm(normal)
        return normal


#-------------------------------------------------------------------------------------------#
# Compound classes (for when a molecule needs to be shifted to origin and then returned)    #
#-------------------------------------------------------------------------------------------#
class ShiftedOperation(Operation):
    """Operations that involve translation before and after the central operation"""

    def __init__(self, shift, operation):
        self.shift = shift
        self.operation = operation

    def __call__(self, data):
        displacement = self.shift.displacement_func(data)
        shift = StaticTranslate(displacement)
        unshift = StaticTranslate(-displacement)

        shift(data)
        static_op = self.operation(data)
        unshift(data)
        return ShiftedOperation(shift, static_op)

#-------------------------------------------------------------------------------------------#
# Main functionality                                                                        #
#-------------------------------------------------------------------------------------------#
class OperationList:
    '''Set of operations that automatically composes appended operations, when possible'''

    def __init__(self):
        self.operations = []

    def append(self, op):
        """add new operation to the list and compose if possible"""
        if len(self) == 0:
            self.operations.append(op)
        elif self[-1].iscomposable(op):
            self[-1].compose(op)
        else:
            self.operations.append(op)

    def __len__(self):
        return len(self.operations)

    def __getitem__(self, key):
        return self.operations[key]

    def __iter__(self):
        return iter(self.operations)


def usage():
    """print usage information"""
    print("Usage:")
    print("  orient [operations]+\n")
    print("File must be in xyz format. Operations can be strung together. Allowed operations are:")
    print("    -t[xyz] <distance>             \t -- translate in x, y, or z direction")
    print("    -ta <atom>                     \t -- translate <atom> to origin")
    print("    -tc                            \t -- translate center of mass to origin")
    print("    -r[xyz] <angle>                \t -- rotate around given axis")
    print("    -rb <angle> <atom> <atom>      \t -- rotate around axis defined by pair of atoms")
    print("    -rp <angle> <a1> <a2> [...]    \t -- rotate around normal of plane defined by list of atoms")
    print("    -rv <angle> <x> <y> <z>        \t -- rotate around defined vector")
    print(
        "    -rd <angle> <a1> <a2> <a3> []  \t -- rotate bond around midpoint of a diene (vector from bond midpoint in direction of normal)"
    )
    print("    -s[xyz]                        \t -- reflect across plane defined by chosen axis as normal")
    print("    -sv                            \t -- reflect across plane defined by specified normal")
    print("    -sb <a1> <a2>                  \t -- reflect across a bond")
    print("    -sp <a1> <a2> <a3> [...]       \t -- reflect across plane fitted to specified atoms")
    print(
        "    -a <atom1> <atom2> <atom3>     \t -- align such that atom1 and atom2 lie along the x-axis and atom3 is in the xy-plane"
    )
    print(
        "    -p <atom1> ... <atomk>         \t -- align such that input atoms form best fit xy-plane and atom1 and atom2 lie along x-axis"
    )
    print("    -op                            \t -- translate to center of mass, orient along principle axes")


def consume_arguments(arguments, geom):
    """consume arguments and return a list of operations"""
    options = arguments[:]
    ops = OperationList()

    while options:
        opt = options.pop(0)
        if opt[1] == 't':  # translations
            if len(opt) != 3:
                raise Exception("Need to specify a translation option (x, y, z, a)")
            trans = None
            if opt[2] in "xyz":
                tr = np.zeros(3)
                tr["xyz".index(opt[2])] = float(options.pop(0))
                trans = StaticTranslate(tr)
            elif opt[2] == 'a':
                trans = AtomTranslate(int(options.pop(0)) - 1, -1.0)
            elif opt[2] == 'c':
                trans = COMTranslate(geom)
            else:
                raise Exception("Unrecognized translation option")

            ops.append(trans)
        elif opt[1] == 'r':  # rotations
            angle = float(options.pop(0))

            if len(opt) != 3:
                raise Exception("Need to specify a rotation option (x, y, z, p, v)")
            axis = np.zeros(3)
            if opt[2] in "xyz":
                axis["xyz".index(opt[2])] = 1.0
                rotate = StaticRotate.from_axis_angle(axis, angle)
            elif opt[2] == 'b':  # atom Pairs
                iatom = int(options.pop(0)) - 1
                jatom = int(options.pop(0)) - 1

                trans = CentroidTranslate([iatom, jatom], -1.0)
                rot = AtomPairRotate(iatom, jatom, angle)

                rotate = ShiftedOperation(trans, rot)
            elif opt[2] == 'p':
                atomlist = []
                try:
                    while len(options) > 0 and options[0][0] != "-":
                        ia = int(options[0]) - 1
                        atomlist.append(ia)
                        options.pop(0)
                except ValueError:
                    pass

                trans = CentroidTranslate(atomlist, -1.0)
                rot = NormalRotate(atomlist, angle)

                rotate = ShiftedOperation(trans, rot)
            elif opt[2] == 'v':  # vector
                axis = np.array([float(options.pop(0)), float(options.pop(0)), float(options.pop(0))])
                rotate = StaticRotate.from_axis_angle(axis, angle)
            elif opt[2] == 'd':
                atomlist = []
                atomlist.append(int(options.pop(0)) - 1)
                atomlist.append(int(options.pop(0)) - 1)
                try:
                    while len(options) > 0 and options[0][0] != "-":
                        ia = int(options[0]) - 1
                        atomlist.append(ia)
                        options.pop(0)
                except ValueError:
                    pass

                trans = CentroidTranslate(atomlist[0:2], -1.0)
                rot = NormalRotate(atomlist, angle)

                rotate = ShiftedOperation(trans, rot)
            else:
                raise Exception("Unrecognized rotation option")

            ops.append(rotate)
        elif opt[1] == 's':  # reflections
            if len(opt) != 3:
                raise Exception("Specify Reflection option (x, y, z, p)")
            if opt[2] in "xyz":
                normal = np.zeros(3)
                normal["xyz".index(opt[2])] = 1.0

                reflect = StaticReflect(normal)
            elif opt[2] == "b":
                iatom = int(options.pop(0)) - 1
                jatom = int(options.pop(0)) - 1

                trans = CentroidTranslate([iatom, jatom], -1.0)
                ref = BondReflect(iatom, jatom)

                reflect = ShiftedOperation(trans, ref)
            elif opt[2] == "v":
                normal = np.array([float(options.pop(0)), float(options.pop(0)), float(options.pop(0))])
                reflect = StaticReflect(normal)
            elif opt[2] == "p":
                iatoms = []
                try:
                    while len(options) > 0 and options[0][0] != "-":
                        ia = int(options[0]) - 1
                        iatoms.append(ia)
                        options.pop(0)
                except ValueError:
                    pass

                trans = CentroidTranslate(iatoms, -1.0)
                ref = PlaneReflect(iatoms)

                reflect = ShiftedOperation(trans, ref)
            else:
                raise Exception("Unrecognized reflection option")

            ops.append(reflect)
        elif opt[1] == 'a':
            # align is called with -a <atom> <atom> <atom>
            ia = int(options.pop(0)) - 1
            ja = int(options.pop(0)) - 1
            ka = int(options.pop(0)) - 1

            ops.append(CentroidTranslate([ia, ja], -1.0))
            ops.append(AlignRotate(ia, ja, ka))
        elif opt[1:] == 'op':
            ops.append(COMTranslate(geom))
            ops.append(InertiaRotate(geom))
        elif opt[1:] == 'p':
            iatoms = []
            try:
                while len(options) > 0 and options[0][0] != "-":
                    ia = int(options[0]) - 1
                    iatoms.append(ia)
                    options.pop(0)
            except ValueError:
                pass

            ops.append(CentroidTranslate(iatoms, -1.0))
            ops.append(PlaneRotate(iatoms))
        else:
            raise Exception("Unknown operation")

    return ops


def orient(arglist):
    if len(arglist) == 0:
        usage()
        return []

    # map from option to number of expected arguments
    nargs = {
        "tc": 0,
        "tx": 1,
        "ty": 1,
        "tz": 1,
        "ta": 1,
        "rx": 1,
        "ry": 1,
        "rz": 1,
        "rb": 3,
        "ra": 3,
        "rp": "+",
        "rv": 4,
        "rd": "+",
        "a": 3,
        "op": 0,
        "p": "+",
        "sx": 0,
        "sy": 0,
        "sz": 0,
        "sv": 3,
        "sb": 2,
        "sp": "+"
    }

    # lets preprocess the options so we let the filename be anywhere in the list
    options = []
    filenames = []
    i = 0
    while True:
        if i == len(arglist):
            break
        op = arglist[i]
        if op[0] != "-":  # maybe a filename
            filenames.append(op)
            i += 1
        else:
            if op[1:] in nargs:
                if "+" == nargs[op[1:]]:
                    narg = 1
                    try:
                        while i + narg < len(arglist) and arglist[i + narg][0] != "-":
                            int(arglist[i + narg])
                            narg += 1
                    except ValueError:
                        pass

                    options.extend(arglist[i:i + narg])
                    i += narg
                else:
                    narg = nargs[op[1:]] + 1
                    options.extend(arglist[i:i + narg])
                    i += narg
            else:
                raise Exception(f"Unrecognized command: \"{op}\" !")

    geoms = []
    for fil in filenames:
        geoms.extend(read_file(fil))

    for g in geoms:
        ops = consume_arguments(options, g)
        for op in ops:
            g.apply(op)
            if DEBUG:
                g.print()

    return geoms


if __name__ == "__main__":

    if len(sys.argv) == 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        usage()
        sys.exit()

    geoms = orient(sys.argv[1:])
    for g in geoms:
        g.print()
