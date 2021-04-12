# SPDX-License-Identifier: GPL-2.0-or-later
# Graphs of filter properties
# Copyright (C) 2021  Trent Piepho <tpiepho@gmail.com>
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

import tg

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
mpl.use('agg')
fgcolor = tg.fgcolor()
mpl.rc('text', color=fgcolor)
mpl.rc('axes', labelcolor=fgcolor) # edgecolor=fgcolor
mpl.rc('xtick', color=fgcolor)
mpl.rc('ytick', color=fgcolor)

# Make a graph title for the HPF graph with f0, Fs, Q in it
def maketitle(f0, fs, Q):
    if Q == np.sqrt(0.5): Q=r'\sqrt{½}'
    return f'Gain and Group Delay: $f_0$ = {f0/1000:g} kHz, $F_s$ = {fs/1000:g} kHz, $Q = {Q}$'

# Plot a IIR filter's gain and group delay
def plotfilter(b, a, fs=48000, title='Gain and Group Delay'):
    G = np.linspace(1, fs/2, 512, endpoint=False)

    z = signal.freqz(b, a, worN=G, fs=fs)[1]
    gd = signal.group_delay((b,a), G, fs=fs)[1]
    logz = 20*np.log10(np.abs(z))

    # facecolor = background color outside plot, use gtk background color?
    # frameon=False = outside plot is transparent.. blends with gtk background by itself!
    fig, ax = plt.subplots(figsize=(8,6), frameon=False, constrained_layout=True)

    ax.plot(G, logz)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin = max(ymin, -80))
    ax.set_title(title)
    ax.minorticks_on()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")

    ax2 = ax.twinx()
    ax2.minorticks_on()
    ax2.plot(G, gd * 1000000 / fs, 'r-')
    ax2.set_ylabel("Delay (µs)")

    # Draw -3 dB stopband
    passdb = 10*np.log10(0.5)
    ax.axhline(passdb, ls=':', c='m')
    stopband = np.where(np.diff(np.concatenate(([False], logz < passdb, [False]))))[0].reshape(-1, 2)
    G = np.append(G, fs/2)
    for s,e in stopband:
        ax.axvspan(G[s], G[e], facecolor='m', alpha=0.10)

    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    plt.close(fig)
    return buffer
