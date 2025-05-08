#!/usr/bin/env python3

import argparse
import os
import simexpal
import subprocess

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=dir_path, required=True, help="The output directory for the binary graph data")

args = parser.parse_args()
output_dir = os.path.abspath(args.output)

config = simexpal.config_for_dir()
build_revision = next(config.all_dev_builds())

dev_build = [build for build in config.all_dev_builds() if build.name == 'graphtool'][0]

for instance in config.all_instances():
    if not instance.check_available():
        print("Instance '{}' is not available.".format(instance.unique_filename))
        continue

    input_path = instance.fullpath
    output_name = instance.unique_filename + '.data'
    output_path = os.path.join(output_dir, output_name)

    if os.path.isfile(output_path):
        print("Instance '{}' already converted.".format(instance.unique_filename))

        try:
            os.symlink(output_path, input_path + '.data')
        except:
            print('Could not establish symbolic link!')
            
        continue

    extra_args = instance.extra_args
    is_directed = '--directed' in extra_args
    is_weighted = '--weighted' in extra_args
    has_timestamps = '--timestamped' in extra_args

    has_file_type = '--file-type' in extra_args
    file_type = 'edges'
    if has_file_type:
        file_type = extra_args[extra_args.index('--file-type') + 1]

    cmd = [
        './graphtool',
        '-i', input_path,
        '-o', output_path
    ]

    if has_file_type:
        cmd.append('--file-type')
        cmd.append(file_type)

    if is_directed:
        cmd.append('--directed')

    if is_weighted:
        cmd.append('--weighted')
    
    if has_timestamps:
        cmd.append('--timestamped')

    print('Converting {}.'.format(instance.shortname))
    subprocess.check_output(cmd, cwd=dev_build.compile_dir)
    print('Finished converting {}.'.format(instance.shortname))

    try:
        os.symlink(output_path, input_path + '.data')
    except:
        print('Could not establish symbolic link!')
