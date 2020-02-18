# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2020 by ExopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Task used to save data to an HDF5 file. This task automatically
detects when it is called inside of (one or more) nested loops and
reshapes the data accordingly. It also saves the sweeped parameters as
a 1D array instead of duplicating them many times.

"""
import os
import errno
import logging
import numbers
import warnings
from inspect import cleandoc
from collections import OrderedDict

import numpy as np
import h5py
from atom.api import Unicode, Enum, Value, Bool, Dict, Int, Typed, List, set_default

from exopy.tasks.api import SimpleTask, RootTask, validators
from exopy.tasks.tasks.logic.loop_task import LoopTask
from exopy.utils.atom_util import ordered_dict_from_pref, ordered_dict_to_pref
from exopy.utils.traceback import format_exc


class SmartSaveTask(SimpleTask):
    """Save the specified data in a HDF5 file.

    Try to detect if it is place inside of nested loops and reshape
    the data accordingly.

    """
    #: Folder in which to save the data.
    folder = Unicode('{default_path}').tag(pref=True, fmt=True)

    #: Name of the file in which to write the data.
    filename = Unicode().tag(pref=True, fmt=True)

    #: Currently opened file object. (File mode)
    file_object = Value()

    #: Header to write at the top of the file.
    header = Unicode().tag(pref=True, fmt=True)

    #: Values to save as an ordered dictionary.
    saved_values = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                    ordered_dict_from_pref))

    #: Parameters to save
    saved_parameters = Typed(OrderedDict, ()).tag(pref=(ordered_dict_to_pref,
                                                        ordered_dict_from_pref))

    #: Data type (float16, float32, etc.)
    datatype = Enum('float16', 'float32', 'float64').tag(pref=True)

    #: Flag indicating whether or not initialisation has been performed.
    initialized = Bool(False)

    database_entries = set_default({'file': None})

    wait = set_default({'activated': True})  # Wait on all pools by default.

    def perform(self):
        """ Collect all data and write them to file.

        """
        # Initialisation.
        if not self.initialized:
            self._formatted_labels = []

            full_folder_path = self.format_string(self.folder)
            filename = self.format_string(self.filename)
            full_path = os.path.join(full_folder_path, filename)
            try:
                self.file_object = h5py.File(full_path, 'w')
            except IOError:
                log = logging.getLogger()
                msg = "In {}, failed to open the specified file."
                log.exception(msg.format(self.name))
                self.root.should_stop.set()

            self.root.resources['files'][full_path] = self.file_object

            f = self.file_object

            parameters_group = f.create_group("parameters")
            for l, v in self.saved_parameters.items():
                label = self.format_string(l)
                value = self.format_and_eval_string(v)
                parameters_group.create_dataset(label, data=value)

            data_group = f.create_group("data")
            for l, v in self.saved_values.items():
                label = self.format_string(l)
                self._formatted_labels.append(label)
                value = self.format_and_eval_string(v)
                if isinstance(value, np.ndarray):
                    names = value.dtype.names
                    if names:
                        for m in names:
                            data_shape = tuple(self._dims) + value[m].shape
                            data_group.create_dataset(label + '_' + m,
                                             data_shape,
                                             self.datatype,
                                             compression="gzip")
                    else:
                        data_shape = tuple(self._dims) + value.shape
                        data_group.create_dataset(label,
                                         data_shape,
                                         self.datatype,
                                         compression="gzip")
                else:
                    data_shape = tuple(self._dims)
                    data_group.create_dataset(label, data_shape,
                                     self.datatype, compression="gzip")
            f.attrs['header'] = self.format_string(self.header)
            f.attrs['count_calls'] = 0
            parameters_group.attrs['parameters_order'] = list(reversed(self._loop_paths.keys()))

            self.initialized = True

        f = self.file_object
        count_calls = f.attrs['count_calls']

        labels = self._formatted_labels
        for i, v in enumerate(self.saved_values.values()):
            value = self.format_and_eval_string(v)
            index = np.unravel_index(count_calls, self._dims)
            if isinstance(value, np.ndarray):
                names = value.dtype.names
                if names:
                    for m in names:
                        f['data'][labels[i] + '_' + m][index] = value[m]
                else:
                    f['data'][labels[i]][index] = value
            else:
                f['data'][labels[i]][index] = value

        f.attrs['count_calls'] = count_calls + 1

    def check(self, *args, **kwargs):
        """Check that all the parameters are correct.

        """
        self.detect_loops()

        for name, path in self._loop_paths.items():
            self._dims.append(self.database.get_value(path, f"{name}_point_number"))
        self._dims.reverse()

        err_path = self.get_error_path()
        test, traceback = super(SmartSaveTask, self).check(*args, **kwargs)
        try:
            full_folder_path = self.format_string(self.folder)
            filename = self.format_string(self.filename)
        except Exception:
            return test, traceback

        full_path = os.path.join(full_folder_path, filename)

        overwrite = False
        if os.path.isfile(full_path):
            overwrite = True
            traceback[err_path + '-file'] = \
                cleandoc('''File already exists, running the measure will
                override it.''')

        try:
            f = open(full_path, 'ab')
            f.close()
            if not overwrite:
                os.remove(full_path)
        except Exception as e:
            mess = 'Failed to open the specified file : {}'.format(e)
            traceback[err_path] = mess.format(e)
            return False, traceback

        labels = set()
        for i, (l, v) in enumerate(self.saved_values.items()):
            try:
                labels.add(self.format_string(l))
            except Exception:
                traceback[err_path + '-label_' + str(i)] = \
                    'Failed to evaluate label {}:\n{}'.format(l, format_exc())
                test = False
            try:
                self.format_and_eval_string(v)
            except Exception:
                traceback[err_path + '-entry_' + str(i)] = \
                    'Failed to evaluate entry {}:\n{}'.format(v, format_exc())
                test = False

        if len(labels) != len(self.saved_values):
            traceback[err_path] = "All labels must be different."
            return False, traceback

        labels = set()
        for i, (l, v) in enumerate(self.saved_parameters.items()):
            try:
                labels.add(self.format_string(l))
            except Exception:
                traceback[err_path + '-label_' + str(i)] = \
                    'Failed to evaluate label {}:\n{}'.format(l, format_exc())
                test = False
            try:
                self.format_and_eval_string(v)
            except Exception:
                traceback[err_path + '-entry_' + str(i)] = \
                    'Failed to evaluate entry {}:\n{}'.format(v, format_exc())
                test = False

        if not test:
            return test, traceback

        if len(labels) != len(self.saved_parameters):
            traceback[err_path] = "All labels must be different."
            return False, traceback

        return test, traceback

    def detect_loops(self):
        n = self.parent
        tmp = []
        self._loop_paths = OrderedDict()
        while not isinstance(n, RootTask):
            if isinstance(n, LoopTask):
                if n.name not in self.saved_parameters:
                    tmp.append((n.name, f"{{{n.name}_loop_values}}"))
                else:
                    tmp.append((n.name, self.saved_parameters[n.name]))
                self._loop_paths[n.name] = n.path
            n = n.parent
        self.saved_parameters = OrderedDict(tmp)

    #: List of the formatted names of the entries.
    _formatted_labels = List()

    #: List of the paths of the loops
    _loop_paths = Typed(OrderedDict, ())

    #: Data dimensions:
    _dims = List()
