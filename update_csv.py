#!/usr/bin/env python
"""
Functions for updating the CSV files that define which probes are allocated
to which objects. It can be run from the command line directly, but the
current instructions for observers tell them to use it from within python.
Dependencies were kept to a minimum so that it could be run on old systems
at the AAT, but in practice this hasn't made any difference.

The definitions of the frames are kept separately, so that the sami package
can be imported on systems that don't have Tkinter installed without error.
Of course, this module cannot be used without Tkinter.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

tkinter_available = True
try:
    import tkinter
    import tkMessageBox
    from update_csv_frames import AllocationEntry
except ImportError:
    print("Warning: Tkinter not available; "
          "probe allocation cannot be done here.")
    tkinter_available = False

astropy_available = True
try:
    import astropy.coordinates
    import astropy.units
except ImportError:
    astropy_available = False

def allocate(infile, outfile, do_central=False, do_object=True, do_sky=True,
             do_guide=True, zero_rotations=False, blank_rotations=False):
    """Main function for interactive allocations.
    Typically called from command line."""
    if not tkinter_available:
        print("Tkinter isn't installed on this computer!")
        print("You can't run the allocation script here!")
        return
    # Set up Tkinter
    root = Tkinter.Tk()
    # Withdraw the Tk window while potential error messages are resolved
    root.withdraw()
    try:
        csvfile = CSV(infile)
    except IOError:
        tkMessageBox.showerror('Error', 'Input file not found!')
        return
    if os.path.isfile(outfile):
        cont = tkMessageBox.askokcancel(
            'Warning', 'Output file already exists. Overwrite?')
        if not cont:
            return
    # Bring the Tk window back again
    root.deiconify()
    # Run the interactive allocator
    csvfile.update_allocation_interactive(do_central=do_central,
                                          do_object=do_object,
                                          do_sky=do_sky,
                                          do_guide=do_guide,
                                          root=root)
    # Done with the interactive part, so destroy the Tkinter interface
    root.destroy()
    # Final tweak to CSV file contents, set the rotations correctly
    if zero_rotations:
        csvfile.zero_rotations()
    elif not blank_rotations:
        csvfile.flip_hexabundles()
    # Make the new CSV file, unless it was cancelled
    if csvfile.ok:
        csvfile.print_contents(outfile)
    return


class CSV:
    """Container for contents of a .csv file."""

    def __init__(self, infile):
        # Dump all the contents of the file into a list
        self.contents = read_file(infile)
        self.central = []
        self.object = []
        self.sky = []
        self.guide = []
        self.unknown = []
        # Remove the newline characters from the contents
        for line_no, line in enumerate(self.contents):
            self.contents[line_no] = remove_newline(line)
        for line_no, line in enumerate(self.contents):
            # First extract the basic data concerning the field
            if line.startswith('LABEL,'):
                self.label = retrieve_parameter(line)
                self.field = self.label.split(' ')[-1]
            elif line.startswith('PLATEID,'):
                self.plateid = retrieve_parameter(line)
            elif line.startswith('UTDATE,'):
                self.utdate = retrieve_parameter(line)
            elif line.startswith('UTTIME,'):
                self.uttime = retrieve_parameter(line)
            elif line.startswith('CENTRE,'):
                self.centre = retrieve_parameter(line)
            elif line.startswith('EQUINOX,'):
                self.equinox = retrieve_parameter(line)
            elif line.startswith('WLEN,'):
                self.wlen = retrieve_parameter(line)
            elif line.startswith('#Name'):
                self.columns = line[1:].split(',')
                # Everything following this is a table, that will be dealt with
                # in the next loop
                break
        line_no += 1
        for line_no, line in enumerate(self.contents[line_no:], line_no):
            # Turn the line into a dictionary with keys given by column headers
            data = dict(zip(self.columns, line.split(',')))
            data['line_no'] = line_no
            # Append the data dictionary to the relevant list of targets
            if 'Type' in data:
                if data['Type'] == 'F':
                    self.central.append(data)
                elif data['Type'] == 'P':
                    self.object.append(data)
                elif data['Type'] == 'S':
                    self.sky.append(data)
                elif data['Type'] == 'G':
                    self.guide.append(data)
                else:
                    self.unknown.append(data)
            else:
                self.unknown.append(data)
        # Set the default titles for different windows
        self.title_list = {'central':[self.label, self.plateid, 'Central probes'],
                           'object':[self.label, self.plateid, 'Object probes'],
                           'sky':[self.label, self.plateid, 'Sky probes'],
                           'guide':[self.label, self.plateid, 'Guide probes']}
        # Store a list of the valid target types
        self.target_type_list = ['central', 'object', 'sky', 'guide']

        return

    def update_allocation_interactive(self, do_central=False, do_object=True,
                                      do_sky=True, do_guide=True, root=None):
        """Update the probe allocations in an interactive manner."""

        # Set up Tkinter, if not already done
        if root is None:
            self.root = Tkinter.Tk()
        else:
            self.root = root

        self.ok = True

        # Store which target types will be examined
        do_target = {'central':do_central, 'object':do_object,
                     'sky':do_sky, 'guide':do_guide}

        # Make a list of all the entry and check windows to be produced
        window_list = []
        for target_type in self.target_type_list:
            if (do_target[target_type] and
                len(self.target_type_to_list(target_type)) > 0):
                window_list.append({'target_type':target_type,
                                    'check':False,
                                    'initial_values':False})
        for target_type in self.target_type_list:
            if (do_target[target_type] and
                len(self.target_type_to_list(target_type)) > 0):
                window_list.append({'target_type':target_type,
                                    'check':True,
                                    'initial_values':False})
        # Need to reverse the list to make best use of pop()
        window_list.reverse()
        self.window_list = window_list
        # Now make the windows one by one (change to "while self.window_list"?)
        while len(self.window_list) > 0:
            self.next_window()
            
        if root is None:
            # I made the root, so it's my responsibility to destroy it
            self.root.destroy()

        return

    def next_window(self):
        """Make the next window in the saved list."""
        if len(self.window_list) != 0:
            window = self.window_list.pop()
            dialog = AllocationEntry(window['target_type'],
                                     self,
                                     window['initial_values'],
                                     master=self.root,
                                     check=window['check'])
            dialog.mainloop()
            dialog.destroy()
        return

    def zero_rotations(self):
        """Set all rotation values to 0.0"""
        for target_type in self.target_type_list:
            self.update_values('Rotation', target_type, 0.0)
        return

    def flip_hexabundles(self):
        """Set all hexabundle rotation values to 180.0"""
        self.update_values('Rotation', 'object', 180.0)
        return
                
    def update_values(self, parameter, target_type, value_list):
        """Set a list of values (automatically converted to strings)."""
        target_list = self.target_type_to_list(target_type)
        # Copy the input value list and pad with the last value
        if isinstance(value_list, list):
            extended_value_list = value_list[:]
        else:
            extended_value_list = [value_list]
        # This looks inefficient
        while len(extended_value_list) < len(target_list):
            extended_value_list.append(extended_value_list[-1])
        # Find where exactly this parameter is stored
        parameter_pos = self.columns.index(parameter)
        for index in range(len(target_list)):
            value = str(extended_value_list[index])
            # First update the value in the target dict
            target_list[index][parameter] = value
            # Then update the value in the contents list
            line_no = target_list[index]['line_no']
            columns = self.contents[line_no].split(',')
            columns[parameter_pos] = value
            self.contents[line_no] = ','.join(columns)
        return

    def get_values(self, parameter, target_type):
        """Return the values of a parameter for one set of targets.
        Attempts to get the type right by trying different options.
        """
        target_list = self.target_type_to_list(target_type)
        value_list_str = []
        for target in target_list:
            value_list_str.append(target[parameter])
        value_list = []
        try:
            # First try to convert to float
            for value_str in value_list_str:
                value_list.append(float(value_str))
        except ValueError:
            # That didn't work, try int next
            try:
                for value_str in value_list_str:
                    value_list.append(int(value_str))
            except ValueError:
                # That also didn't work, leave as str
                value_list = list(value_list_str)
        return value_list

    def ra(self, target_type='object', decimal=True):
        """Return a list of RA values for a given target type.
        If decimal==True, returns decimal degrees, otherwise strings
        in the same format as the .csv file."""
        ra_str = self.get_values('RA', target_type)
        if decimal:
            if not astropy_available:
                print('Error: astropy required to convert to decimal')
                return None
            ra_decimal = []
            for hours in ra_str:
                ra_decimal.append(astropy.coordinates.RA(
                    hours, unit=astropy.units.hour).degrees)
            return ra_decimal
        else:
            return ra_str

    def dec(self, target_type='object', decimal=True):
        """Return a list of Dec values for a given target type.
        If decimal==True, returns decimal degrees, otherwise strings
        in the same format as the .csv file."""
        dec_str = self.get_values('Dec', target_type)
        if decimal:
            if not astropy_available:
                print('Error: astropy required to convert to decimal')
                return None
            dec_decimal = []
            for degrees in dec_str:
                dec_decimal.append(astropy.coordinates.Dec(
                    degrees, unit=astropy.units.degree).degrees)
                ### BUG IN ASTROPY v0.2.0!!! ###
                if '-' in degrees and dec_decimal[-1] > 0:
                    dec_decimal[-1] *= -1
            return dec_decimal
        else:
            return dec_str

    def print_contents(self, file_out, set_readwrite=True):
        """Write the contents to a new file."""
        bad_lines = [item['line_no']
                     for target_type in self.target_type_list
                     for item in self.target_type_to_list(target_type)
                     if item['Probe'] == '']
        output = [line for line_no, line in enumerate(self.contents)
                  if line_no not in bad_lines]
        dir_out = os.path.dirname(file_out)
        if dir_out != '' and not os.path.exists(dir_out):
            os.makedirs(dir_out)
        f_out = open(file_out, 'w')
        f_out.write('\n'.join(output) + '\n')
        f_out.close()
        if set_readwrite:
            os.chmod(file_out, 0o666)
        return

    def target_type_to_list(self, target_type):
        """Convert a target type string to the relevant list of targets."""
        return {'central':self.central,
                'object':self.object,
                'sky':self.sky,
                'guide':self.guide}[target_type]




def remove_newline(line):
    """Remove (\r)\n from the end of a string and return it."""
    if line.endswith('\n'):
        line = line[:-1]
    if line.endswith('\r'):
        line = line[:-1]
    return line

def retrieve_parameter(line):
    """Return the parameter value from one line in a .csv file."""
    return line.split(',', 1)[1]
    

def read_file(infile):
    """Read an ASCII file and return the contents."""
    f_in = open(infile)
    contents = f_in.readlines()
    f_in.close()
    return contents


def repeated_elements(s):
    """Return True if any elements in s are repeated. Not used?"""
    if len(s) <= 1:
        return False

    for i in range(len(s) - 1):
        if s[i] in s[i+1:]:
            return True

    return False

# If run from the command line, do enter_allocation_order
if __name__ == "__main__":
    description = """Update a SAMI .csv allocation file.
    By default, the hexabundles, sky fibres and guide probes are allocated
    but the central hole is not; and the hexabundles are set to a rotation
    of 180.0."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile', help='Input .csv file')
    parser.add_argument('outfile', help='Output .csv file')
    parser.add_argument('-z', '--zerorotations', action='store_true',
                        help='Set all rotation values to 0.0')
    parser.add_argument('-b', '--blankrotations', action='store_true',
                        help='Do not set anything in rotation values')
    parser.add_argument('-c', '--central', action='store_true',
                        help='Allocate central probes')
    parser.add_argument('--noobject', action='store_true',
                        help='Do not allocate object probes')
    parser.add_argument('--nosky', action='store_true',
                        help='Do not allocate sky probes')
    parser.add_argument('--noguide', action='store_true',
                        help='Do not allocate guide probes')
    args = parser.parse_args()

    allocate(args.infile, args.outfile, do_central=args.central,
             do_object=(not args.noobject), do_sky=(not args.nosky),
             do_guide=(not args.noguide),
             zero_rotations=args.zerorotations,
             blank_rotations=args.blankrotations)
    
