# This script reads a galaxy data-set and generates galaxy images

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
 
# Save galaxy plot
def save_galaxy_plot(output_path, par_pos_a, par_pos_b):
        fig = plt.figure(facecolor='black')
        ax = plt.gca()
        plt.scatter(par_pos_a, par_pos_b, color='white', marker='o', lw=0, s=1)
        plt.ylim([-40, 40])
        plt.xlim([-40, 40])
        ax.set_aspect('equal', 'datalim')
        ax.set(xlabel='r', ylabel='r')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi = 10, facecolor=fig.get_facecolor())
        plt.close(fig)


# Read galaxy information (id_group, id_subgroup, is_elliptical, no_particles,
# positions) and generate galaxy images

input_file_path = 'galaxies.dat'
output_dir_path = 'galaxy_images/'
validation_set_prop = 0.2

with open(input_file_path, 'rb') as gal_file:

    count = 0
    while True:

        # Check if there are more galaxies
        aux = gal_file.read(4)
        if aux == b'':
            exit()

        # Read a galaxy from the binary file
        id_group = int.from_bytes(aux,byteorder='little')
        id_subgroup = int.from_bytes(gal_file.read(4),byteorder='little')
        is_elliptical = int.from_bytes(gal_file.read(2),byteorder='little')
        no_particles = int.from_bytes(gal_file.read(8),byteorder='little')
        par_pos_x = np.zeros(no_particles)
        par_pos_y = np.zeros(no_particles)
        par_pos_z = np.zeros(no_particles)
        for i in range(no_particles):
            par_pos_x[i] = struct.unpack('d', gal_file.read(8))[0]
        for i in range(no_particles):
            par_pos_y[i] = struct.unpack('d', gal_file.read(8))[0]
        for i in range(no_particles):
            par_pos_z[i] = struct.unpack('d', gal_file.read(8))[0]

        # Generate and save galaxy plots

        if np.random.uniform() < validation_set_prop:
            subdir = 'validation/'
        else:
            subdir = 'train/'

        if is_elliptical:
            subdir = subdir + 'Elliptical/' 
        else:
            subdir = subdir + 'Non-Elliptical/' 

        output_path = output_dir_path+subdir+str(id_group)+"_"+str(id_subgroup)+"_xy_"+str(is_elliptical)
        save_galaxy_plot(output_path, par_pos_x, par_pos_y)

        output_path = output_dir_path+subdir+str(id_group)+"_"+str(id_subgroup)+"_xz_"+str(is_elliptical)
        save_galaxy_plot(output_path, par_pos_x, par_pos_z)

        output_path = output_dir_path+subdir+str(id_group)+"_"+str(id_subgroup)+"_yz"+str(is_elliptical)
        save_galaxy_plot(output_path, par_pos_y, par_pos_z)

        print("Image No.:+"count"+. Galaxy Id:", id_group,"-", id_subgroup, "... Completed.")
        count += 1 









