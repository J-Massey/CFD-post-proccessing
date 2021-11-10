# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia, J. Massey
@description: Input/output module to read binary Fortran dat of 2D and 3D fields with different
    number of components. Also useful to unpack text dat in column-written.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import vtk
import pickle
from postproc.calc import make_periodic


# Functions
def read_data(file, shape, **kwargs):
    """
    Return the velocity components of a velocity vector field stored in binary format.
    The dat field is supposed to have been written as: (for k; for j; for i;) where the last dimension
    is the quickest varying index. Each record should have been written as: u, v, w.
    The return velocity components are always converted in np.double precision type.
    :param file: fn to read from.
    :param shape: Shape of the dat as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
    :param kwargs:
        dtype: numpy dtype object. Single or double precision expected.
        stream (depracated, use always stream output): type of access of the binary output. If false, there is a 4-byte header
            and footer around each "record" in the binary fn (means +2 components at each record) (can happen in some
            Fortran compilers if access != 'stream').
        periodic: If the user desires to make the dat periodic in a certain direction: (0,0,1) makes periodic in z.
        ncomponents: Specify the number of components. Default = ndims of the field
    :return: the components of the vector or scalar field.
    """
    dtype = kwargs.get('dtype', np.single)
    periodic = kwargs.get('periodic', (0,0,0))
    ncomponents = kwargs.get('ncomponents', len(shape))

    shape = tuple(reversed(shape))
    shape_comp = shape + (ncomponents,)
    period_axis = [i for i, pc in enumerate(periodic) if pc]

    f = open(file, 'rb')
    data = np.fromfile(file=f, dtype=dtype).reshape(shape_comp)
    f.close()

    if len(shape) == 2:
        if ncomponents == 1:
            u = data[:, :, 0].transpose(1, 0)
            u = u.astype(np.float64, copy=False)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
            return u
        elif ncomponents == 2:
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
                    v = make_periodic(v, axis=ax)
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return u, v
        elif ncomponents == 3:
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            w = data[:, :, 2].transpose(1, 0)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
                    v = make_periodic(v, axis=ax)
                    w = make_periodic(w, axis=ax)
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return u, v, w
        else:
            raise ValueError("Number of components is not <=3")
    elif len(shape) == 3:
        if ncomponents == 1:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
            u = u.astype(np.float64, copy=False)
            return u
        elif ncomponents == 2:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
                    v = make_periodic(v, axis=ax)
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return u, v
        elif ncomponents == 3:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            w = data[:, :, :, 2].transpose(2, 1, 0)
            del data
            if any(periodic):
                for ax in period_axis:
                    u = make_periodic(u, axis=ax)
                    v = make_periodic(v, axis=ax)
                    w = make_periodic(w, axis=ax)
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return u, v, w
        else:
            raise ValueError("Number of components is not <=3")
    else:
        raise ValueError("Shape is not two- nor three-dimensional")


def read_data_raw(file, shape, ncomponents):
    shape = tuple(reversed(shape))
    shape_comp = shape + (ncomponents,)

    f = open(file, 'rb')
    data = np.fromfile(file=f, dtype=np.single).reshape(shape_comp)
    f.close()
    return data


def read_and_write_fractioned_mean_data(f_w_list, shape, **kwargs):
    """
    Computes a weighted average of files which containg partial averages of quantities.
    :param f_w_list: list of tuples containing (fn, weight).
    :param shape: Shape of the dat as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
    :param kwargs:
    :param kwargs:
        dtype: numpy dtype object. Single or double precision expected.
        stream (depracated, use always stream output): type of access of the binary output. If false, there is a 4-byte header
            and footer around each "record" in the binary fn (means +2 components at each record) (can happen in some
            Fortran compilers if access != 'stream').
        periodic: If the user desires to make the dat spanwise periodic (true) or not (false).
        ncomponents: Specify the number of components. Default = ndims of the field
    :return: The weighted average of the files containing partial averages.
    """
    global a, file, b, c
    dtype = kwargs.get('dtype', np.single)
    ncomponents = kwargs.get('ncomponents', len(shape))
    if ncomponents == 1:
        aw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
            aw_tot += a*w
            w_tot += w

        a_mean = aw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)

        a_mean.tofile(file[:-4]+'_python_mean.dat')
        return

    elif ncomponents == 2:
        aw_tot = 0
        bw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
                b = data[:, :, 1]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
                b = data[:, :, :, 1]
            aw_tot += a*w
            bw_tot += b*w
            w_tot += w

        a_mean, b_mean = aw_tot/w_tot, bw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)
        b_mean = b_mean.astype(np.float64, copy=False)
        r_mean = np.stack((a_mean, b_mean), axis=-1)
        r_mean.tofile(file[:-4]+'_python_mean.dat')
        return

    elif ncomponents == 3:
        aw_tot = 0
        bw_tot = 0
        cw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
                b = data[:, :, 1]
                c = data[:, :, 2]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
                b = data[:, :, :, 1]
                c = data[:, :, :, 2]
            aw_tot += a * w
            bw_tot += b * w
            cw_tot += c * w
            w_tot += w

        a_mean, b_mean, c_mean =  aw_tot/w_tot, bw_tot/w_tot, cw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)
        b_mean = b_mean.astype(np.float64, copy=False)
        c_mean = c_mean.astype(np.float64, copy=False)
        r_mean = np.stack((a_mean, b_mean, c_mean), axis=-1)

        f_root = '/'.join(file.split('/')[0:-3])+'/'
        f_name = file.split('/')[-1][:-4]
        r_mean.tofile(f_root + 'output/' +f_name + '_python_mean.dat')
        return
    else:
         raise ValueError("Number of components is not <=3")


def unpack2Dforces(file, D=1, U=1):
    """
    Unpacks ASCII files coontaining the following columns: non-dimensional time, time step, fx, fy, vx, vy.
    fx, fy are the pressure forces and vx, vy the viscous forces (not returned with this function).
    :param file: File to read from.
    :param D: Characteristic length.
    :param U: characteristic velocity.
    :return: time, fx, fy.
    """
    tD, dt, fx, fy, _, _ = np.loadtxt(file, unpack=True)  # 2D
    return tD*D/U, fx, fy


def unpack3Dforces(file, D=1, U=1):
    """
    Unpacks ASCII files containing the following columns: non-dimensional time, time step, fx, fy, fz, vx, vy, vz.
    fx, fy, fz are the pressure forces and vx, vy, vz the viscous forces (not returned with this function).
    :param file: File to read from.
    :param D: Characteristic length.
    :param U: characteristic velocity.
    :return: time, fx, fy.
    """
    tD, dt, fx, fy, _, _, _, _ = np.loadtxt(file, unpack=True) #3D
    return tD*D/U, fx, fy


def unpack_flex_forces(file, names):
    """
    Unpacks ASCII files containing columns of user defined parameters. This allows the testing of new functions in
    the body module and flexibility in what is printed to your 'fort.9' fn
    :param file: File to read from.
    :param names: dtype=list The names of the parameters you've decided to write in the simulation
    :return: New arrays assigned to your names
    """
    try:
        names = np.loadtxt(file, unpack=True)
    except IndexError:
        print("Get the number of entries right you fucking mug")
    return names


def unpackTimeSeries(file, npoints):
    """
    Unpacks ASCII files containing the following columns: non-dimensional time, point1, point2, ...
    :param file:
    :param npoints: number of points recorded in the fn
    :return: time, point1, point2, ...
    """
    if npoints == 1:
        t, p = np.loadtxt(file, unpack=True)  # 3D
        return t, p
    elif npoints == 2:
            t, p1, p2 = np.loadtxt(file, unpack=True)  # 3D
            return t, p1, p2
    elif npoints == 3:
            t, p1, p2, p3 = np.loadtxt(file, unpack=True)  # 3D
            return t, p1, p2, p3
    else:
        raise ValueError("Number of points is not <=3")


def readTimeSeries(file):
    """
    Reads ASCII files containing the following columns: non-dimensional time, point1, point2, ...
    :param file:
    :return: 2D numpy array. Each column is a time series. Normally time is the first column. torch = a[:,0]
    """
    return np.loadtxt(file)


def readPressureArcLength(file):
    p, L = np.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
    return p, L


def importExpCpTheta(file):
    theta, cp = np.loadtxt(file, unpack=True, delimiter=',')
    return theta, cp


def read_vti(file):
    reader = vtk.vtkXMLPImageDataReader()
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()
    pointData = data.GetPointData()

    sh = data.GetDimensions()[::-1]
    ndims = len(sh)

    # get vector field
    v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims,))
    vec = []
    for d in range(ndims):
        a = v[..., d]
        vec.append(a)
    # get scalar field
    sca = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1,))

    # Generate grid
    # nPoints = dat.GetNumberOfPoints()
    (xmin, xmax, ymin, ymax, zmin, zmax) = data.GetBounds()
    grid3D = np.mgrid[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]

    return np.transpose(np.array(vec), (0, 3, 2, 1)), np.transpose(sca, (0, 3, 2, 1)), grid3D


def read_vtr(fn):
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    reader = vtk.vtkXMLPRectilinearGridReader()
    reader.SetFileName(fn)
    reader.Update()
    data = reader.GetOutput()
    pointData = data.GetPointData()
    sh = data.GetDimensions()[::-1]
    ndims = len(sh)

    # get vector field
    try:
        v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims,))
        vec = []
        for d in range(ndims):
            a = np.array(v[..., d])
            vec.append(a)
        vec = np.array(vec)
        # get scalar field
        sca = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1,))

        # get grid
        x = np.array(data.GetXCoordinates())
        y = np.array(data.GetYCoordinates())
        z = np.array(data.GetZCoordinates())

        return np.transpose(vec, (0, 3, 2, 1)), np.transpose(sca, (0, 3, 2, 1)), np.array((x, y, z))
    except ValueError:
        print('\n' + fn + ' corrupt, skipping for now')


def rotate_2D_flow_field(fn, length_scale, rotation=0):
    """
    Rotates and scales vtr file
    Args:
        fn: The path to the 'datp' folder
        length_scale: length scale of the simulation
        rotation: Rotate the grid. If you're running a simulation with
                  an angle of attack, it's better to rotate the flow than
                  the foil because of the meshing.

    Returns: X, Y - coordinates (useful for indexing)
             U, V - rotated velocity components
             w    - un-rotated z velocity component
             p    - pressure field

    """
    rot = rotation / 180 * np.pi
    data = read_vtr(fn)
    # Get the grid
    x, y, z = data[2]
    X, Y = np.meshgrid(x / length_scale, y / length_scale)
    X = np.cos(rot) * X + np.sin(rot) * Y
    Y = -np.sin(rot) * X + np.cos(rot) * Y

    u, v, w = data[0]
    U = np.cos(rot) * u + np.sin(rot) * v
    V = -np.sin(rot) * u + np.cos(rot) * v
    p = data[1]
    p = np.reshape(p, [np.shape(p)[0], np.shape(p)[2], np.shape(p)[3]])
    return X, Y, U, V, w, p


def pvd_parser(fn, n=None):
    times = []
    files = []
    with open(fn) as f:
        lines = f.readlines()

    for line in lines[2:-2][:n]:
        l = line.split(" ")
        times.append(float(l[1].split("\"")[1]))
        files.append(str(l[4].split("\"")[1]))
    return times, files


def write_object(obj, fname):
    with open(fname, 'wb') as output:  # Overwrites any existing fn.
        pickle.dump(obj, output, protocol=pickle.HIGHEST_PROTOCOL)
    return


def read_object(fname):
    with open(fname, 'rb') as input:  # Overwrites any existing fn.
        obj = pickle.load(input)
    return obj


def read_txt(fname, skiprows=0, delimiter=','):
    lines = np.loadtxt(fname, skiprows=skiprows, delimiter=delimiter, unpack=True)
    return lines