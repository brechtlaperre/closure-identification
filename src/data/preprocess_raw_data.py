r'''Preprocess raw datafiles into inputfiles

Executing this file creates an .h5 file for each experiment stored in the data/raw folder.
The files corresponding to each experiment are stored in an .h5 file with the following structure

experiment_name.h5
    |- attrs
        |-- xcells, ycells, zcells
        |-- dx, dy
    |- timestamp
        |- species
            |-- features
            |-- attr: dims (List)

@author: Brecht Laperre
'''
import os
from collections import namedtuple
import numpy as np
import pandas as pd
import h5py
import argparse

def read_preprocessed_data(resolution, timestamp, species, file):
    '''Read datafile returned from preprocess.py
    Input:
        resolution: experiment name
        timestamp: simulation snapshot time
        species: particle species
        file: filehandel
    Output:
        dataset: Pandas DataFrame. Row correspond with features at single cell
        dims: Dimensions of the domain. xcells x ycells x zcells
        centers: Location of reconnection
    '''
    dims = []
    centers = []
    sets = []
    dataset = None
    if str(timestamp) in file.keys():
        subdata = file['{}/{}'.format(timestamp, species)]
        subkeys = subdata.keys()
        skeys = [k for k in subkeys]
        if type(file['{}/{}/{}'.format(timestamp, species, skeys[0])]) is h5py.Group:
            for regionkey in subkeys:
                regiondata = subdata['regionkey']
                set = pd.DataFrame()
                for k in regiondata.keys():
                    set[k] = np.array(regiondata[k])
                set['id'] = '{}_{}_{}'.format(timestamp, regionkey, resolution)
                dims.append(regiondata.attrs['dim'])
                centers.append(regiondata.attrs['center'])
                sets.append(set)
            dataset = pd.concat(sets, ignore_index=True)
        else:
            dataset = pd.DataFrame()
            for k in subkeys:
                dataset[k] = np.array(subdata[k])
            dataset['id']=timestamp
            dims = subdata.attrs['dim']
            centers = None

    return dataset, dims, centers

def read_variables(data, coords, spec_keys, species, field_keys):
    '''Extract features from the raw datafiles
    This function opens the raw h5py data and takes data at z = 0. 
    The data over the full domain is gathered in a dictionary. The /Step#0/Block/ part is removed from the feature name.

    The result is the raw data in a more readable dictionary and more usefull format 
    '''
    basekey = '/Step#0/Block'
    read_vars = {'X': coords.X.flatten(), 'Y': coords.Y.flatten() }
    
    if type(species) is not list:
        species = [species]
    
    for sp_k in spec_keys:
        read_vars[sp_k] = 0
        for sp in species:
            read_vars[sp_k] += data['{}/{}_{}/0'.format(basekey, sp_k, sp)][0, :, :].flatten()
        
    for f in field_keys:
        read_vars[f] = data['{}/{}/0'.format(basekey, f)][0, :, :].flatten()

    df = pd.DataFrame.from_dict(read_vars)

    return df

def read_raw_data_from_file(file, grid_size, attr_group, values, fields):
    try:
        raw_data = h5py.File(file, 'r')
    except OSError:
        print('File not found: {}'.format(file))
        return None 
    print('Processing file {}'.format(file))
    
    # Retrieve dimensions from data
    zcells, ycells, xcells = raw_data['/Step#0/Block/Pxx_0/0'].shape
    # print(raw_data['/Step#0/Block/'].keys())

    GridStep = namedtuple('GridStep', ['dx', 'dy', 'dz'])
    dx = grid_size.Lx / (xcells - 1)
    dy = grid_size.Ly / (ycells - 1)
    dz = grid_size.Lz / (zcells - 1)
    gridstep = GridStep(dx=dx, dy=dy, dz=dz)

    attr_group.attrs.create('xcells', xcells)
    attr_group.attrs.create('ycells', ycells)
    attr_group.attrs.create('zcells', zcells-1)

    attr_group.attrs.create('dx', dx)
    attr_group.attrs.create('dy', dy)

    x_range = np.linspace(0, grid_size.Lx, xcells, endpoint=True)
    y_range = np.linspace(0, grid_size.Ly, ycells, endpoint=True)
    z_range = np.linspace(0, grid_size.Lz, zcells-1, endpoint=True)

    _, Y, X = np.meshgrid(z_range, y_range, x_range, indexing='ij')
    Coords = namedtuple('Coords', ['X', 'Y'])
    coords = Coords(X=X, Y= Y)
    Nserie = coords.X.shape[0]*coords.X.shape[1]*coords.X.shape[2]
    dims = coords.X.shape[1:]

    # read information from the electrons. Use this to compute agyro for all the species of this snapshot
    raw = read_variables(raw_data, coords, values, [0, 2], fields)
    agyro = compute_agyro(raw['Bx'], raw['By'], raw['Bz'],
                                  raw['Pxx'], raw['Pyy'], raw['Pzz'], 
                                  raw['Pxy'], raw['Pxz'], raw['Pyz'], 
                                  raw['Jx'], raw['Jy'], raw['Jz'], 
                                  raw['N'], -256)
    return raw_data, agyro, coords, gridstep, Nserie, dims


def search_dir(path):
    names = []
    for root, _, files in os.walk(path):
        print(root)
        for file_ in files:
            if 'DoubleHarris-Fields' in file_:
                names.append(path + '/' + file_)

    return names

def store_data(file, dataset, Nserie):
    ''' Given a h5py file, this function stores the columns of a dataset to the file - IMPLICIT
    Input:
        file: open h5py file
        dataset: pandas dataset
        Nserie: length of the dataset
     '''
    # file is an h5py file
    for key in dataset.keys():
        if dataset[key].dtype == 'object':
                dtype = "S5" # Set to string of fixed length
        else:
            dtype = dataset[key].dtype
        s = file.require_dataset('{}'.format(key), (Nserie,), dtype=dtype)
        s[...] = dataset[key].values
        
def compute_agyro(Bx, By, Bz, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, Jx, Jy, Jz, N, qom):
    '''Compute agyrotropy
    '''
    small = 1e-10
    
    small=qom /np.abs(qom)*1e-10 
    Pxx = (Pxx - Jx*Jx / (N+small) ) /qom
    Pyy = (Pyy - Jy*Jy / (N+small) ) /qom
    Pzz = (Pzz - Jz*Jz / (N+small) ) /qom
    Pxy = (Pxy - Jx*Jy / (N+small) ) /qom
    Pxz = (Pxz - Jx*Jz / (N+small) ) /qom
    Pyz = (Pyz - Jy*Jz / (N+small) ) /qom
    
    Nserie = len(Bx)

    P = np.zeros((Nserie, 3, 3))
    P[:, 0, 0] = Pxx
    P[:, 0, 1] = Pxy
    P[:, 0, 2] = Pxz
    P[:, 1, 1] = Pyy
    P[:, 1, 2] = Pyz
    P[:, 2, 2] = Pzz
    P[:, 1, 0] = P[:, 0, 1]
    P[:, 2, 0] = P[:, 0, 2]
    P[:, 2, 1] = P[:, 1, 2]
    
    B = np.zeros((Nserie, 3))
    B[:, 0] = Bx
    B[:, 1] = By
    B[:, 2] = Bz
    
    B = B/(small + np.sum(B**2, axis=1)[:, np.newaxis])
    
    # Scudder
    
    N1 = np.zeros(P.shape)
    for l in range(3):
        N1[:,:,l] = np.cross(B, P[:,l])
    
    N = np.zeros(P.shape)
    for l in range(3):
        N[:,l,:] = np.cross(B, N1[:,l,:])
    
    
    #alpha = N[:,0,0] + N[:,1,1] + N[:,2,2] + small
    #beta = -(N[:,0,1]**2 + N[:,0,2]**2 + N[:,1,2]**2 - N[:,0,0]*N[:,1,1] - N[:,0,0]*N[:,1,1] - N[:,1,1]*N[:,2,2])

    #Agyro = 2*np.sqrt(alpha**2-4*beta)/alpha

    lamb = np.sort(np.linalg.eig(N)[0])
    
    Agyro = 2*np.abs(lamb[:, 2] - lamb[:, 1])/(lamb[:, 2] + lamb[:, 1])
    
    if Agyro.dtype == np.complex128:
        print(np.iscomplex(Agyro).sum(), ' out of ', len(Agyro))

    #if Agyro_2.dtype == np.complex128:
    #    print('Agyro_2: {} out of {}'.format(np.iscomplex(Agyro_2).sum(), len(Agyro_2)))

    return Agyro

def compute_temperature_and_thermal_velocity(P, rho, qom):
    ''' Compute pressure components normalized by the density
    Computes the full pressure tensor from the velocity, pressure and density in the raw data
    Output: target now containing the values {Pxx, Pxy, Pxz, Pyx, Pyy, Pzz, Psc, T}
    '''
    small = 1e-10
    Psc = (P['Pxx'].values + P['Pyy'].values + P['Pzz'].values)/3
    T = Psc / np.abs(rho + small)
    vth = np.sqrt(Psc / np.abs(rho+small) * np.abs(qom))

    return Psc, T, vth


def compute_pressure(Bx, By, Bz, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, Jx, Jy, Jz, N, qom):
    ''' Compute parallel and perpendicular pressure components from pressure tensor
    '''
    small=qom /np.abs(qom)*1e-10 
    # Compute corrected pressure tensor components
    Pxx = (Pxx - Jx*Jx / (N+small) ) /qom
    Pyy = (Pyy - Jy*Jy / (N+small) ) /qom
    Pzz = (Pzz - Jz*Jz / (N+small) ) /qom
    Pxy = (Pxy - Jx*Jy / (N+small) ) /qom
    Pxz = (Pxz - Jx*Jz / (N+small) ) /qom
    Pyz = (Pyz - Jy*Jz / (N+small) ) /qom

    b2D = 1e-10 + Bx *Bx + By *By 
    b = b2D + Bz *Bz # b = ||B||^2

    # Pressure tensor component in parallel direction
    PPAR = Bx *Pxx *Bx + 2*Bx *Pxy *By + 2*Bx *Pxz *Bz 
    PPAR += By *Pyy *By + 2*By *Pyz *Bz 
    PPAR += Bz *Pzz *Bz 
    PPAR = PPAR /b 
    
    newman = True
    if(newman): 

        # Perpendicular 1 normal vector, normalized
        Perp1x = By/np.sqrt(b2D)
        Perp1y = -Bx/np.sqrt(b2D)
        Perp1z = 0

        # Checks out
        PPER1 = By *Pxx *By - 2*By *Pxy *Bx + Bx *Pyy *Bx 
        PPER1 = PPER1 /b2D 

        # Perpendicular 2 normal vector, normalized
        Perp2x = Bz *Bx /np.sqrt(b *b2D) 
        Perp2y = Bz *By /np.sqrt(b *b2D) 
        Perp2z = -np.sqrt(b2D /b)

        # Checks out
        PPER2 = Perp2x *Pxx *Perp2x + 2*Perp2x *Pxy *Perp2y + 2*Perp2x *Pxz *Perp2z 
        PPER2 += Perp2y *Pyy *Perp2y + 2*Perp2y *Pyz *Perp2z 
        PPER2 += Perp2z *Pzz *Perp2z 

        # Off-diagonal for parallel and first perpendicular direction
        PParP1 = Bx*Pxx*Perp1x + Bx*Pxy*Perp1y + Bx*Pxz*Perp1z
        PParP1 += By*Pxy*Perp1x + By*Pyy*Perp1y + By*Pyz*Perp1z
        PParP1 += Bz*Pxz*Perp1x + Bz*Pyz*Perp1y + Bz*Pzz*Perp1z
        PParP1 = PParP1 / np.sqrt(b)

        # Off-diagonal for parallel and second perpendicular direction
        PParP2 = Bx*Pxx*Perp2x + Bx*Pxy*Perp2y + Bx*Pxz*Perp2z
        PParP2 += By*Pxy*Perp2x + By*Pyy*Perp2y + By*Pyz*Perp2z
        PParP2 += Bz*Pxz*Perp2x + Bz*Pyz*Perp2y + Bz*Pzz*Perp2z
        PParP2 = PParP2 / np.sqrt(b)

        # Off-diagonal for both perpendicular directions
        PPER12 = Perp1x*Pxx*Perp2x + Perp1x*Pxy*Perp2y + Perp1x*Pxz*Perp2z
        PPER12 += Perp1y*Pxy*Perp2x + Perp1y*Pyy*Perp2y + Perp1y*Pyz*Perp2z
        PPER12 += Perp1z*Pxz*Perp2x + Perp1z*Pyz*Perp2y + Perp1z*Pzz*Perp2z

    else:

        # unit vector along Jperp = J - Jpar
        Perp1x = Jx - Jx * Bx /b # ?? add Bx/b term?
        Perp1y = Jy - Jy * By /b # ?? add By/b term?
        Perp1z = Jz - Jz * Bz /b # ?? ddd Bz/b term?
        Perp1n = np.sqrt(Perp1x **2 + Perp1y **2 + Perp1z **2) 
        Perp1x = Perp1x  /Perp1n 
        Perp1y = Perp1y  /Perp1n 
        Perp1z = Perp1z  /Perp1n 
        
        # Pressure tensor component in first perpendicular direction
        PPER1 = Perp1x *Pxx *Perp1x + 2*Perp1x *Pxy *Perp1y + 2*Perp1x *Pxz *Perp1z 
        PPER1 += Perp1y *Pyy *Perp1y + 2*Perp1y *Pyz *Perp1z 
        PPER1 += Perp1z *Pzz *Perp1z 

        # unit vector b x pepr1
        Perp2x = (By  * Perp1z - Bz  * Perp1y)/b 
        Perp2y = (Bz  * Perp1x - Bx  * Perp1z)/b 
        Perp2z = (Bx  * Perp1y - By  * Perp1x)/b 
        
        # Pressure tensor component in second perpendicular direction
        PPER2 = Perp2x *Pxx *Perp2x + 2*Perp2x *Pxy *Perp2y + 2*Perp2x *Pxz *Perp2z 
        PPER2 += Perp2y *Pyy *Perp2y + 2*Perp2y *Pyz *Perp2z 
        PPER2 += Perp2z *Pzz *Perp2z 

        # Off-diagonal for parallel and first perpendicular direction
        PParP1 = Bx*Pxx*Perp1x + Bx*Pxy*Perp1y + Bx*Pxz*Perp1z
        PParP1 += By*Pxy*Perp1x + By*Pyy*Perp1y + By*Pyz*Perp1z
        PParP1 += Bz*Pxz*Perp1x + Bz*Pyz*Perp1y + Bz*Pzz*Perp1z
        PParP1 = PParP1 / np.sqrt(b)

        # Off-diagonal for parallel and second perpendicular direction
        PParP2 = Bx*Pxx*Perp2x + Bx*Pxy*Perp2y + Bx*Pxz*Perp2z
        PParP2 += By*Pxy*Perp2x + By*Pyy*Perp2y + By*Pyz*Perp2z
        PParP2 += Bz*Pxz*Perp2x + Bz*Pyz*Perp2y + Bz*Pzz*Perp2z
        PParP2 = PParP2 / np.sqrt(b)

        # Off-diagonal for both perpendicular directions
        PPER12 = Perp1x*Pxx*Perp2x + Perp1x*Pxy*Perp2y + Perp1x*Pxz*Perp2z
        PPER12 += Perp1y*Pxy*Perp2x + Perp1y*Pyy*Perp2y + Perp1y*Pyz*Perp2z
        PPER12 += Perp1z*Pxz*Perp2x + Perp1z*Pyz*Perp2y + Perp1z*Pzz*Perp2z

    P_xyz = pd.DataFrame(np.stack([Pxx, Pyy, Pzz, Pxy, Pxz, Pyz], axis=1), columns=['Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz'])
    P_magn = pd.DataFrame(np.stack([PPAR, PPER1, PPER2, PParP1, PParP2, PPER12], axis=1), columns=['Ppar', 'Pper1', 'Pper2', 'Pparp1', 'Pparp2', 'Pper12'])

    #return Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, PPAR, PPER1, PPER2, PParP1, PParP2, PPER12
    return P_xyz, P_magn

def compute_gradients(fx, fy, fz, dx, dy, dims, varname):
    '''Compute the gradients of the velocity and the magnetic field
    Making use of the numpy gradient function ( f(i+1) - f(i-1))/2dx )
    https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    ''' 
    Dxx, Dxy = np.gradient(fx.to_numpy().reshape(dims), dx, dy)
    Dyx, Dyy = np.gradient(fy.to_numpy().reshape(dims), dx, dy)
    Dzx, Dzy = np.gradient(fz.to_numpy().reshape(dims), dx, dy)
    dfxdx, dfxdy = Dxx.flatten(), Dxy.flatten()
    dfydx, dfydy = Dyx.flatten(), Dyy.flatten()
    dfzdx, dfzdy = Dzx.flatten(), Dzy.flatten()
    
    return {'d{}xdx'.format(varname): dfxdx, 'd{}xdy'.format(varname): dfxdy, 
            'd{}ydx'.format(varname): dfydx, 'd{}ydy'.format(varname): dfydy, 
            'd{}zdx'.format(varname): dfzdx, 'd{}zdy'.format(varname): dfzdy}

def compute_heatflux(P, Qx, Qy, Qz, Jx, Jy, Jz, rho, qom):
    ''' Compute the heatflux according to the equations discussed with Gianni
    
    '''
    small = 1e-10

    # Compute bulk energy flux 
    Ubulk = 0.5*(Jx**2 + Jy**2 + Jz**2)/(rho+small)/qom

    Qbulkx = Jx/(rho+small)*Ubulk
    Qbulky = Jy/(rho+small)*Ubulk
    Qbulkz = Jz/(rho+small)*Ubulk

    # Thermal energy density
    Uth = 0.5*(P['Pxx']+P['Pyy']+P['Pzz'])

    # Enthalpy flux
    Qenthx = Jx/(rho+small) * Uth + (Jx  * P['Pxx'] + Jy * P['Pxy'] + Jz * P['Pxz'])  /(rho+small)
    Qenthy = Jy/(rho+small) * Uth + (Jx  * P['Pxy'] + Jy * P['Pyy'] + Jz * P['Pyz'])  /(rho+small)
    Qenthz = Jz/(rho+small) * Uth + (Jx  * P['Pxz'] + Jy * P['Pyz'] + Jz * P['Pzz'])  /(rho+small)
    
    # Finally, compute heatflux
    Qhfx = Qx - Qbulkx - Qenthx
    Qhfy = Qy - Qbulky - Qenthy
    Qhfz = Qz - Qbulkz - Qenthz

    return Qhfx, Qhfy, Qhfz

def compute_magnetic_crossterms(Bx, By, Bz):
    '''Compute crossterms of the magnetic field
    '''
    Bxy = Bx*By
    Bxz = Bx*Bz
    Byz = By*Bz

    return Bxy, Bxz, Byz
            
def compute_velocity_field(Jx, Jy, Jz, rho):
    '''Compute velocity fields from the current
    '''
    small = 1e-10
    rho = rho + small

    Vx = Jx/rho
    Vy = Jy/rho
    Vz = Jz/rho

    Vxy = Vx*Vy
    Vxz = Vx*Vz
    Vyz = Vy*Vz

    return Vx, Vy, Vz, Vxy, Vxz, Vyz

def compute_mhd_electric_field(Bx, By, Bz, Vx, Vy, Vz):
    ''' Compute electric field from the MHD case
    E = - (v x B)
    '''
    Ex_mhd = - ( (Vy * Bz) - (Vz * By))
    Ey_mhd = - ( (Vz * Bx) - (Vx * Bz))
    Ez_mhd = - ( (Vx * By) - (Vy * Bx))

    return Ex_mhd, Ey_mhd, Ez_mhd

def process_raw_data_folder(folder_name):
    '''The script looks at folder in 'data/raw/{folder_name}', with folder name specified in ftypes below.
    The script then processess everything and dumps it in a file in 'data/processed/processed_{folder_name}.h5
    The script 'src/data/process.py' prepares the processed data to be used in ML experiments
    '''
    source = 'data/raw'
    target = 'data/processed'

    fields = ['Bx', 'By', 'Bz', 'divB', 'divE', 'Ex', 'Ey', 'Ez']

    values = ['Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz', 'Jx', 'Jy', 'Jz', 'rho', 'EFx', 'EFy', 'EFz', 'N']

    # There are 4 species, 0 and 2 are electrons, 1 and 3 are ions. 
    species = [0, 3, 2, '0+2'] # only consider electrons for now #[0, 1, 2, 3, [0, 2], [1, 3]]
    sp_names = {0: 0, 1: 1, 2: 2, 3: 3, '0+2': [0,2], '1+3': [1,3]} # [0, 1, 2, 3, '0+2', '1+3']
    qom_sp = {0: -256.0, 1: 1.0, 2: -256.0, 3: 1.0, '0+2': -256.0, '1+3': 1.0} # Charge over mass ratio of each species. Ions are normalized to 1
    
    processed_data = h5py.File('{}/processed_data_{}.h5'.format(target, folder_name), 'w')

    attr_group = processed_data.create_group('attrs')
    files = search_dir(source + '/' + folder_name)

    ### See SimulationData.txt
    GridSize = namedtuple('GridSize', ['Lx', 'Ly', 'Lz'])
    grid_size = GridSize(30, Ly=40, Lz=0.1)

    # Set background magnetic field. Values found in simulation input file
    if folder_name == 'high_res' or folder_name == 'high_res_2':
        B_bg = {'x': 0, 'y': 0, 'z': 0.00096}
    elif folder_name == 'high_res_hbg':
        B_bg = {'x': 0, 'y': 0, 'z': 0.0096}
    elif folder_name == 'high_res_bg3':
        B_bg = {'x': 0.0, 'y': 0, 'z': 3.0}
    else:
        B_bg = {'x': 0.0, 'y': 0, 'z': 0.0}
    

    for file in files:
        # Check if chosen file exists and can be opened  
        raw_data, agyro, coords, gridstep, Nserie, dims = read_raw_data_from_file(file, grid_size, attr_group, values, fields)
        # Read timestamp from filename
        t = int(file.split('_')[-1].split('.')[0])
        time_group = processed_data.create_group(str(t))

        for sp in species:
            # Compute the necessary values from the raw data
            qom = qom_sp[sp]

            spec_group = time_group.create_group('Species_{}'.format(sp)) # Define a group for the species
            spec_group.attrs.create('qom', qom)
            raw = read_variables(raw_data, coords, values, sp_names[sp], fields) # Take raw_data and put it in a more useable format
            # Remove background magnetic field

            proc_df = pd.DataFrame({'X': coords.X.flatten(), 'Y': coords.Y.flatten()}) # copy cell coordinates
            # Compute the pressure
            P_xyz, P_magn = compute_pressure(raw['Bx'], raw['By'], raw['Bz'], 
                                                raw['Pxx'], raw['Pyy'], raw['Pzz'], 
                                                raw['Pxy'], raw['Pxz'], raw['Pyz'],
                                                raw['Jx'], raw['Jy'], raw['Jz'], 
                                                raw['N'], qom)

            # Compute scalar pressure, temperature and thermal velocity
            Psc, T, vth = compute_temperature_and_thermal_velocity(P_xyz, raw['rho'], qom)
            
            # Compute heatflux
            Qhfx, Qhfy, Qhfz = compute_heatflux(P_xyz, 
                                                raw['EFx'], raw['EFy'], raw['EFz'], 
                                                raw['Jx'], raw['Jy'], raw['Jz'], 
                                                raw['rho'], qom)
            
            # Remove background magnetic field
            Bx = raw['Bx']# - B_bg['x']
            By = raw['By']# - B_bg['y']
            Bz = raw['Bz']# - B_bg['z']

            # Compute crossterms
            Bxy, Bxz, Byz = compute_magnetic_crossterms(raw['Bx'], raw['By'], raw['Bz'])
            Vx, Vy, Vz, Vxy, Vxz, Vyz = compute_velocity_field(raw['Jx'], raw['Jy'], raw['Jz'], raw['rho'])

            # Compute Lorentz electric field
            Ex_mhd, Ey_mhd, Ez_mhd = compute_mhd_electric_field(raw['Bx'], raw['By'], raw['Bz'], Vx, Vy, Vz)

            # Compute gradients
            dB = compute_gradients(Bx, By, Bz, gridstep.dx, gridstep.dy, dims, 'B')
            dV = compute_gradients(Vx, Vy, Vz, gridstep.dx, gridstep.dy, dims, 'V')

            dB_df = pd.DataFrame().from_dict(dB)
            dV_df = pd.DataFrame().from_dict(dV)                

            # store density
            proc_df['rho'] = np.abs(raw['rho'].values).reshape(Nserie, 1)
            proc_df['Agyro'] = agyro
            
            proc_df = pd.concat([proc_df, P_xyz], axis=1)
            proc_df = pd.concat([proc_df, P_magn], axis=1)
            
            proc_df['Jx'] = raw['Jx']
            proc_df['Jy'] = raw['Jy']
            proc_df['Jz'] = raw['Jz']

            proc_df['Bx'] = Bx
            proc_df['By'] = By
            proc_df['Bz'] = Bz
            proc_df['Bxy'] = Bxy
            proc_df['Bxz'] = Bxz
            proc_df['Byz'] = Byz
            proc_df['B_magn'] = np.sqrt(Bx**2 + By**2 + Bz**2)
            proc_df['alpha'] = proc_df['rho']**3/proc_df['B_magn']**2

            proc_df['Vx'] = Vx
            proc_df['Vy'] = Vy
            proc_df['Vz'] = Vz
            proc_df['Vxy'] = Vxy
            proc_df['Vxz'] = Vxz
            proc_df['Vyz'] = Vyz

            proc_df['Ex_mhd'] = Ex_mhd
            proc_df['Ey_mhd'] = Ey_mhd
            proc_df['Ez_mhd'] = Ez_mhd

            proc_df['Ex'] = raw['Ex'] - Ex_mhd
            proc_df['Ey'] = raw['Ey'] - Ey_mhd
            proc_df['Ez'] = raw['Ez'] - Ez_mhd

            proc_df['HFx'] = Qhfx
            proc_df['HFy'] = Qhfy
            proc_df['HFz'] = Qhfz

            proc_df['Psc'] = Psc
            proc_df['T'] = T
            proc_df['vth'] = vth

            proc_df = pd.concat([proc_df, dB_df, dV_df], axis=1)
            spec_group.attrs.create('dim', dims)

            store_data(spec_group, proc_df, np.prod(list(dims)))

        raw_data.close()
    processed_data.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Raw data parser')
    parser.add_argument('--folder_name',  '-f',
                        dest="foldername",
                        metavar='string',
                        help =  'name of folder inside data/raw/',
                        default='all')

    args = parser.parse_args()
    
    if args.foldername == 'all':
        standard_folders = ['high_res_bg0', 'high_res_hbg', 'high_res_2', 'high_res_bg3'] 
        for folder in standard_folders:
            process_raw_data_folder(folder)
    else:
        process_raw_data_folder(args.foldername)
    