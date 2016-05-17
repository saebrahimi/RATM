#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import os

actions = ('boxing', 'handclapping', 'handwaving',
           'jogging', 'running', 'walking')
persons = range(1, 26)
scenarios = range(1, 5)

nvids = len(actions) * len(persons) * len(scenarios)
i = 0

for action in actions:
    for person in persons:
        for scenario in scenarios:
            i += 1
            print i, '/', nvids
            vid_fname = 'vid/person{0:02d}_{1}_d{2}_uncomp.avi'.format(
                person, action, scenario)
            print vid_fname
            target_folder_name = os.path.join('frames', os.path.splitext(
                os.path.split(vid_fname)[1])[0])
            if os.path.exists(vid_fname):
                os.system('mkdir -p {0}'.format(target_folder_name))
                os.system(('ffmpeg -i {0} -vf scale=160:120 '
                           '{1}/%3d.png').format(vid_fname, target_folder_name))
            else:
                print vid_fname
                os.system('echo {0} >> missing.txt'.format(vid_fname))


# vim: set ts=4 sw=4 sts=4 expandtab:
