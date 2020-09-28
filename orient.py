#!/usr/bin/env python

#
#  Orient molecule: simple commandline xyz manipulation
#  Copyright (C) 2013-2017  Shane M. Parker <smparker@uci.edu>

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

DEBUG = False

# masses of most common isotopes to 3 decimal points, from
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
masses = {
    'x': 0.000,
    'h': 1.008,
    'he': 4.003,
    'li': 7.016,
    'be': 9.012,
    'b': 11.009,
    'c': 12.000,
    'n': 14.003,
    'o': 15.995,
    'f': 18.998,
    'ne': 19.992,
    'na': 22.990,
    'mg': 23.985,
    'al': 26.981,
    'si': 27.977,
    'p': 30.974,
    's': 31.972,
    'cl': 34.969,
    'ar': 39.962,
    'k': 38.964,
    'ca': 39.963,
    'sc': 44.956,
    'ti': 47.948,
    'v': 50.944,
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
    'i': 126.904,
    'xe': 131.904
}

class Geometry(object):
    '''Stores all of the data in an xyz file'''
    def __init__(self, names, coordinates, comment = "", extras = None):
        self.names = names
        self.coordinates = coordinates
        self.natoms = coordinates.shape[0]
        self.comment = comment
        self.extras = extras
        self.com = np.array([])

    def print(self):
        print("%s" % self.natoms)
        print("%s" % self.comment)

        for i in range(self.natoms):
            extra = "   ".join(["%14.10f" % float(x) for x in self.extras[i]]) if self.extras else ""
            print("%3s   %14.10f   %14.10f   %14.10f   %s" % (self.names[i], self.coordinates[i, 0],
                self.coordinates[i, 1], self.coordinates[i, 2], extra))

    def computeCOM(self):
        '''
        Returns the center of mass of the geometry.
        '''
        if (len(self.com) == 3):
            return self.com
        else:
            self.com = np.dot(self.mass, self.coordinates) / np.sum(self.mass)

    def computeInertia(self):
        '''Returns the moment of inertia tensor'''
        com = self.computeCOM()
        data = self.coordinates - com

        inertial_tensor = -np.einsum("ax,a,ay->xy", data, self.mass, data)
        return inertial_tensor

def read_xyz(filename):
    '''reads xyz file and returns Geometry object'''
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
                extra = data[4:]

                names.append(name.capitalize())
                coords.append( [float(x), float(y), float(z)] )
                if extra:
                    extras.append(extra)

            out.append(Geometry(names, np.array(coords), comment=comment, extras=extras))

            line = f.readline()

    return out

class Operation(object):
    '''Base class for generic operation'''
    def __call__(self, data):
        '''act on provided coordinate data'''
        raise Exception("Improper use of Operation class!")

    def iscomposable(self, op):
        '''return true if this op can composed with input op'''
        raise Exception("Improper use of Operation class!")

    def compose(self, op):
        '''compose this op and input op'''
        raise Exception("Improper use of Operation class!")

#-------------------------------------------------------------------------------------------#
# Translation classes                                                                       #
#-------------------------------------------------------------------------------------------#

class Translate(Operation):
    def __call__(self, data):
        displacement = self.displacement_func(data)
        data += np.repeat(displacement.reshape(1,3), data.shape[0], axis=0)

    def iscomposable(self, op):
        '''Safety first'''
        return False

    def compose(self, trans):
        if not isinstance(trans, Translate):
            raise Exception("Improper use of Translate.compose()!")
        else:
            func1, func2 = self.displacement_func, trans.displacement_func
            def new_disp_func(data):
                return func1(data) + func2(data)
            self.displacement_func = new_disp_func

class StaticTranslate(Translate):
    def __init__(self, displacement):
        self.displacement = displacement

    def displacement_func(self, data):
        return self.displacement

    def iscomposable(self, op):
        return isinstance(op, StaticTranslate)

class DynamicTranslate(Translate):
    def iscomposable(self, op):
        return False

class AtomTranslate(DynamicTranslate):
    def __init__(self, iatom, fac = 1.0):
        self.iatom = iatom
        self.fac = fac

    def displacement_func(self, data):
        return self.fac * data[self.iatom,:]

class CentroidTranslate(DynamicTranslate):
    def __init__(self, atomlist, fac = 1.0):
        self.atomlist = atomlist
        self.fac = fac / len(atomlist)

    def displacement_func(self, data):
        return np.sum(data[self.atomlist,:], axis=0) * self.fac

class COMTranslate(DynamicTranslate):
    def __init__(self, geom):
        self.mass = [ masses[n.lower()] for n in geom.names ]
        self.totalmass = sum(self.mass)

    def displacement_func(self, data):
        return -np.dot(self.mass, data) / np.sum(self.mass)

#-------------------------------------------------------------------------------------------#
# Rotation classes                                                                          #
#-------------------------------------------------------------------------------------------#
class Rotate(Operation):
    '''Generic rotation'''
    def __init__(self):
        self.rotate_func = None

    def __call__(self, data):
        A = self.rotate_func(data)
        detA = np.linalg.det(A)
        if detA < 0.0:
            raise Exception("Determinant of Rotation needs to be 1")
        tmp = np.dot(data, A.T)
        data[:] = tmp[:]

    def iscomposable(self, op):
        '''Safety first'''
        return False

    def compose(self, rot):
        if not isinstance(rot, Rotate):
            raise Exception("Improper use of Rotate.compose()")
        else:
            func1, func2 = self.rotate_func, rot.rotate_func
            def new_rotate_func(data):
                A1 = func1(data)
                A2 = func2(data)
                return np.dot(A1, A2)
            self.rotate_func = new_rotate_func

    @staticmethod
    def axis_angle(axis, angle):
        axis /= np.linalg.norm(axis)
        theta = m.radians(angle)

        costheta = m.cos(theta)
        sintheta = m.sin(theta)

        Ex = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.0]])

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

    @classmethod
    def from_axis_angle(cls, axis, angle):
        return cls(Rotate.axis_angle(axis, angle))

class DynamicRotate(Rotate):
    def iscomposable(self, op):
        return False

class AtomPairRotate(DynamicRotate):
    def __init__(self, i, j, angle):
        self.i, self.j = i, j
        self.angle = angle

    def rotate_func(self, data):
        iatom = data[self.i,:]
        jatom = data[self.j,:]
        axis = jatom - iatom

        return Rotate.axis_angle(axis, self.angle)

class AlignRotate(DynamicRotate):
    def __init__(self, i, j, k):
        self.i, self.j, self.k = i, j, k

    def rotate_func(self, data):
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

class InertiaRotate(DynamicRotate):
    def __init__(self, geom):
        self.mass = [ masses[n.lower()] for n in geom.names ]

    def rotate_func(self, data):
        inertial_tensor = -np.einsum("ax,a,ay->xy", data, self.mass, data)
        # negate sign to reverse the sorting of the tensor
        eig, axes = np.linalg.eigh(-inertial_tensor)
        axes = axes.T

        # adjust sign of axes so third moment moment is positive new in X, and Y axes
        testcoords = np.dot(data, axes.T) # a little wasteful, but fine for now
        thirdmoment = np.einsum("ax,a->x", testcoords**3, self.mass)

        for i in range(2):
            if thirdmoment[i] < 1.0e-6:
                axes[i,:] *= -1.0

        # rotation matrix must have determinant of 1
        if np.linalg.det(axes) < 0.0:
            axes[2,:] *= -1.0

        return axes

class NormalRotate(DynamicRotate):
    def __init__(self, atomlist, angle):
        self.atomlist = atomlist
        self.angle = angle

    def rotate_func(self, data):
        atoms = np.array([ data[i,:] for i in self.atomlist ])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2,:]
        normal /= np.linalg.norm(normal)

        # compute plane of first three atoms to specify the direction
        v1 = data[self.atomlist[1],:] - data[self.atomlist[0],:]
        v2 = data[self.atomlist[2],:] - data[self.atomlist[0],:]
        crossnormal = np.cross(v1, v2)

        if np.dot(normal, crossnormal) < 0.0:
            normal *= -1.0

        return Rotate.axis_angle(normal, self.angle)

class PlaneRotate(DynamicRotate):
    def __init__(self, atomlist):
        self.atomlist = atomlist

    def rotate_func(self, data):
        atoms = np.array([ data[i,:] for i in self.atomlist ])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2,:]
        normal /= np.linalg.norm(normal)

        # compute plane of first three atoms to specify the direction
        v1 = data[self.atomlist[1],:] - data[self.atomlist[0],:]
        v2 = data[self.atomlist[2],:] - data[self.atomlist[0],:]
        crossnormal = np.cross(v1, v2)

        if np.dot(normal, crossnormal) < 0.0:
            normal *= -1.0

        xvec = atoms[1,:] - atoms[0,:]
        vec1 = xvec - np.dot(xvec,normal)*normal
        vec1 /= np.linalg.norm(vec1)

        vec2 = np.cross(normal, vec1)

        A = np.array([ vec1, vec2, normal])
        if np.linalg.det(A) < 0.0:
            A[2,:] *= -1.0
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

    def iscomposable(self, op):
        '''Disable composing reflections'''
        return False

    def compose(self, op):
        raise Exception("Improper use of Reflect.compose")

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
        return data[self.jatom,:] - data[self.iatom,:]

class PlaneReflect(Reflect):
    '''Reflect across a plane fitted to a group of atoms
    Undefine results if used away from origin'''
    def __init__(self, atomlist):
        self.atomlist = atomlist

    def reflect_func(self, data):
        atoms = np.array([ data[i,:] for i in self.atomlist ])

        U, s, V = np.linalg.svd(atoms, full_matrices=False)

        normal = V[2,:]
        normal /= np.linalg.norm(normal)
        return normal

#-------------------------------------------------------------------------------------------#
# Compound classes (for when a molecule needs to be shifted to origin and then returned)    #
#-------------------------------------------------------------------------------------------#
class ShiftedOperation(Operation):
    def __init__(self, shift, operation):
        self.shift = shift
        self.operation = operation

    def __call__(self, data):
        displacement = self.shift.displacement_func(data)
        unshift = StaticTranslate(-displacement)

        self.shift(data)
        self.operation(data)
        unshift(data)

    def iscomposable(self, op):
        return False

    def compose(self, op):
        raise Exception("Cannot compose Compound classes")

#-------------------------------------------------------------------------------------------#
# Main functionality                                                                        #
#-------------------------------------------------------------------------------------------#
class OperationList(object):
    '''Set of operations that automatically composes appended operations, when possible'''
    def __init__(self):
        self.operations = []

    def append(self, op):
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
    print("    -rd <angle> <a1> <a2> <a3> []  \t -- rotate bond around midpoint of a diene (vector from bond midpoint in direction of normal)")
    print("    -s[xyz]                        \t -- reflect across plane defined by chosen axis as normal")
    print("    -sv                            \t -- reflect across plane defined by specified normal")
    print("    -sb <a1> <a2>                  \t -- reflect across a bond")
    print("    -sp <a1> <a2> <a3> [...]       \t -- reflect across plane fitted to specified atoms")
    print("    -a <atom1> <atom2> <atom3>     \t -- align such that atom1 and atom2 lie along the x-axis and atom3 is in the xy-plane")
    print("    -p <atom1> ... <atomk>         \t -- align such that input atoms form best fit xy-plane and atom1 and atom2 lie along x-axis")
    print("    -op                            \t -- translate to center of mass, orient along principle axes")

def consume_arguments(arguments, geom):
    options = arguments[:]
    ops = OperationList()

    while(options):
        opt = options.pop(0)
        if (opt[1] == 't'): # translations
            if (len(opt) != 3):
                raise Exception(
                    "Need to specify a translation option (x, y, z, a)")
            trans = None
            if opt[2] in "xyz":
                tr = np.zeros(3)
                tr["xyz".index(opt[2])] = float(options.pop(0))
                trans = StaticTranslate(tr)
            elif (opt[2] == 'a'):
                trans = AtomTranslate(int(options.pop(0))-1, -1.0)
            elif (opt[2] == 'c'):
                trans = COMTranslate(geom)
            else:
                raise Exception("Unrecognized translation option")

            ops.append(trans)
        elif opt[1] == 'r': # rotations
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
                axis = np.array(
                    [float(options.pop(0)), float(options.pop(0)), float(options.pop(0))])
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
        elif (opt[1] == 's'): # reflections
            if len(opt) != 3:
                raise Exception("Specify Reflection option (x, y, z, p)")
            if opt[2] in "xyz":
                normal = np.zeros(3)
                normal["xyz".index(opt[2])] = 1.0

                reflect = StaticReflect(normal)
            elif opt[2] == "b":
                iatom = int(options.pop(0)) - 1
                jatom = int(options.pop(0)) - 1

                trans = CentroidTranslate([iatom,jatom], -1.0)
                ref = BondReflect(iatom, jatom)

                reflect = ShiftedOperation(trans, ref)
            elif opt[2] == "v":
                normal = np.array(
                    [float(options.pop(0)), float(options.pop(0)), float(options.pop(0))])
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
        elif (opt[1] == 'a'):
            # align is called with -a <atom> <atom> <atom>
            ia = int(options.pop(0))-1
            ja = int(options.pop(0))-1
            ka = int(options.pop(0))-1

            ops.append(CentroidTranslate([ia,ja], -1.0))
            ops.append(AlignRotate(ia,ja,ka))
        elif (opt[1:] == 'op'):
            ops.append(COMTranslate(geom))
            ops.append(InertiaRotate(geom))
        elif (opt[1:] == 'p'):
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
        return

    # map from option to number of expected arguments
    nargs = { "tc" : 0, "tx" : 1, "ty" : 1, "tz" : 1, "ta" : 1,
            "rx" : 1, "ry" : 1, "rz" : 1, "rp" : "+", "rb" : 3, "ra": 3, "rp": "+", "rv" : 3, "rd" : "+",
            "a" : 3, "op" : 0, "p" : "+",
            "sx" : 0, "sy" : 0, "sz" : 0, "sv" : 3, "sb" : 2, "sp" : "+" }

    # lets preprocess the options so we let the filename be anywhere in the list
    options = []
    filenames = [ ]
    i = 0
    while True:
        if i == len(arglist):
            break
        op = arglist[i]
        if op[0] != "-": # maybe a filename
            filenames.append(op)
            i += 1
        else:
            if op[1:] in nargs:
                if "+" == nargs[op[1:]]:
                    narg = 1
                    try:
                        while i+narg < len(arglist) and arglist[i+narg][0] != "-":
                            int(arglist[i+narg])
                            narg += 1
                    except ValueError:
                        pass

                    options.extend(arglist[i:i+narg])
                    i += narg
                else:
                    narg = nargs[op[1:]]+1
                    options.extend(arglist[i:i+narg])
                    i += narg
            else:
                raise Exception("Unrecognized command: \"%s\" !" % op)

    geoms = []
    for fil in filenames:
        geoms.extend(read_xyz(fil))

    for g in geoms:
        ops = consume_arguments(options, g)
        for op in ops:
            op(g.coordinates)
            if DEBUG:
                g.print()

    return geoms

if __name__ == "__main__":
    import sys

    if (len(sys.argv) == 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        exit()

    geoms = orient(sys.argv[1:])
    for g in geoms:
        g.print()
