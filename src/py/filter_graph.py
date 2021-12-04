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
def plotfilter(b, a, fs=48000, title='Gain and Group Delay', figsize=(800,600), groupdelay=True):
    G = np.linspace(2, fs/2, 512, endpoint=False)

    z = signal.freqz(b, a, worN=G, fs=fs)[1]
    logz = 20*np.log10(np.abs(z))

    dpi = mpl.rcParams['figure.dpi']
    figsize = tuple(s / dpi for s in figsize)
    # facecolor = background color outside plot, use gtk background color?
    # frameon=False = outside plot is transparent.. blends with gtk background by itself!
    fig, ax = plt.subplots(figsize=figsize, frameon=False, constrained_layout=True)

    ax.plot(G, logz)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin = np.clip(ymin, -80, -1), ymax = max(ymax, 1))
    ax.set_title(title)
    ax.minorticks_on()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB) (blue)")
    ax.axhline(0, lw=0.5, ls='--', c='k')

    if groupdelay:
        gd = signal.group_delay((b,a), G[1:], fs=fs)[1]
        ax2 = ax.twinx()
        ax2.minorticks_on()
        gd = gd * 1000000 / fs # samples to μs
        # Attempt to crop off asymptotes
        p = np.percentile(gd, [1, 99], interpolation='nearest')
        p += np.array([-0.1, 0.1]) * p.ptp()
        r = np.array([gd.min(), gd.max()]).clip(p[0], p[1])
        r += np.array([-1, 1]) * ax2.margins() * r.ptp()
        ax2.set_ylim(r)
        ax2.plot(G[1:], gd, 'r-')
        ax2.set_ylabel("Delay (µs) (red)")

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

def plotfiltern(n, **kwargs):
    flt = tg.getfilter(n)
    return plotfilter(flt[0], flt[1], fs=tg.getsr(), **kwargs)

def plotfilterchain(**kwargs):
    # Turn 2nd order filter chain to a single higher-order transfer function.
    flt = signal.sos2tf(np.reshape(tg.getfilterchain(), (-1,6)))
    return plotfilter(flt[0], flt[1], fs=tg.getsr(), **kwargs)
