"""
Definitions of frames for update_csv. Kept separate to isolate Tkinter.
There is some overlap of business logic and GUI code that could probably
be cleaned up.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import Tkinter
import tkMessageBox
import tkFont

min_central = 1
max_central = float("inf")

min_hexa = 1
max_hexa = 13

min_sky = 1
max_sky = 26

min_guide = 1
max_guide = float("inf")

max_rows = 13

class AllocationEntry(Tkinter.Frame):
    """Master window for entry (or check) of probe allocations."""
    def __init__(self, target_type, csvfile, initial_values,
                 master=None, check=False):
        Tkinter.Frame.__init__(self, master)
        # Store the data provided
        self.target_type = target_type
        self.target_list = csvfile.target_type_to_list(target_type)
        self.title_list = csvfile.title_list[target_type]
        self.csvfile = csvfile
        self.initial_values = initial_values
        self.check = check
        self.probe_type = self.target_list[0]['Type']
        if self.probe_type == 'F':
            self.min_probe = min_central
            self.max_probe = max_central
        elif self.probe_type == 'P':
            self.min_probe = min_hexa
            self.max_probe = max_hexa
        elif self.probe_type == 'S':
            self.min_probe = min_sky
            self.max_probe = max_sky
        elif self.probe_type == 'G':
            self.min_probe = min_guide
            self.max_probe = max_guide
        else:
            # Should raise an exception really
            pass
        self.pack()
        # Make the contents of the window
        self.title_frame = TitleFrame(self.title_list, master=self)
        prefix = ''     # Hangover from H#001 days
        self.boxes_frame = BoxesFrame(prefix, master=self)
        self.buttons_frame = ButtonsFrame(master=self)

class TitleFrame(Tkinter.Frame):
    """Frame containing the titles."""
    def __init__(self, title_list, master=None):
        Tkinter.Frame.__init__(self, master)
        self.pack()
        for title in title_list:
            label = Tkinter.Label(self, text=title)
            label.pack()

class BoxesFrame(Tkinter.Frame):
    """Frame containing the entry boxes and relevant labels."""
    def __init__(self, prefix, master=None):
        Tkinter.Frame.__init__(self, master)
        self.master = master
        self.pack()
        self.create_boxes(prefix)

    def create_boxes(self, prefix):
        validate_command = self.register(self.isdigit)
        self.boxes = []
        self.names = []
        for target_no, target in enumerate(self.master.target_list):
            if target.has_key('Name'):
                name = target['Name']
            else:
                name = 'Target {0}'.format(target_no)
            try:
                field = str(self.master.csvfile.field)
            except AttributeError:
                field = ''
            name = (name + ': ' + field +
                    self.master.probe_type + str(target_no+1))
            self.names.append(name)
            label_text = name + ': ' + prefix
            label = Tkinter.Label(self, text=label_text)
            column, row = divmod(target_no, max_rows)
            label.grid(row=row, column=2*column, sticky='E')
            if self.master.check or self.master.initial_values:
                text = target['Probe']
            else:
                text = ''
            if self.master.check:
                self.boxes.append(Tkinter.Label(
                    self, text=text, font=tkFont.Font(weight=tkFont.BOLD),
                    fg='red'))
            else:
                self.boxes.append(
                    Tkinter.Entry(self, validate='key',
                                  validatecommand=(validate_command, '%S')))
                if self.master.initial_values:
                    self.boxes[-1].insert(0, text)
            self.boxes[-1].grid(row=row, column=2*column+1)
        return

    def isdigit(self, text):
        """Check function for valid input, must be numerical."""
        return text.isdigit()

class ButtonsFrame(Tkinter.Frame):
    """Frame containing ok/cancel or yes/no buttons."""
    def __init__(self, master=None):
        Tkinter.Frame.__init__(self, master)
        self.master = master
        self.pack()
        self.create_buttons()

    def create_buttons(self):
        if self.master.check:
            question = Tkinter.Label(self, text='All allocations OK?')
            question.grid(row=0, column=0, columnspan=2)
            cancel_button = Tkinter.Button(self, text='No', command=self.no)
            ok_button = Tkinter.Button(self, text='Yes', command=self.yes)
            row = 1
        else:
            cancel_button = Tkinter.Button(self, text='Cancel',
                                           command=self.cancel)
            ok_button = Tkinter.Button(self, text='OK', command=self.ok)
            row = 0
        cancel_button.grid(row=row, column=0)
        ok_button.grid(row=row, column=1)
        return

    def ok(self):
        """Called when OK is clicked: check, save and continue."""
        values = []
        for box in self.master.boxes_frame.boxes:
            values.append(box.get())
        redo = False
        # Check for valid probe numbers
        converted_values = []
        for value, name in zip(values, self.master.boxes_frame.names):
            # Check a value has been entered
            if value == '':
                message = ('No probe allocated for ' + name +
                           '. Continue anyway?')
                cont = tkMessageBox.askokcancel('Warning', message)
                if cont:
                    converted_values.append('')
                    continue
                else:
                    redo = True
                    break
            # Check the probe number is valid
            if (int(value) < self.master.min_probe or
                int(value) > self.master.max_probe):
                message = 'Unrecognised probe allocated for ' + name
                tkMessageBox.showerror('Error', message)
                redo = True
                break
            else:
                converted_values.append(value.lstrip('0'))
            # Check we haven't already had this probe
            if len(converted_values) > 1:
                if converted_values[-1] in converted_values[:-1]:
                    message = ('Duplicate allocation for probe number ' +
                               value)
                    tkMessageBox.showerror('Error', message)
                    redo = True
                    break
        if not redo:
            # Everything was ok, save entered values and close window
            self.master.csvfile.update_values(
                'Probe', self.master.target_type, converted_values)
            self.quit()
        return

    def cancel(self):
        """Called when Cancel is clicked: close everything without saving."""
        self.master.csvfile.ok = False
        self.master.csvfile.window_list = []
        self.quit()
        return

    def yes(self):
        """Called when Yes is clicked: close this window and carry on."""
        self.quit()
        return

    def no(self):
        """Called when No is clicked: redo allocation for this target."""
        self.quit()
        self.master.csvfile.window_list.append({'target_type':self.master.target_type,
                                                'check':True,
                                                'initial_values':False})
        self.master.csvfile.window_list.append({'target_type':self.master.target_type,
                                                'check':False,
                                                'initial_values':True})
        return
