# SPDX-License-Identifier: GPL-2.0-or-later
# Spectrograms
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
import libtfr

fgcolor = tg.fgcolor()
mpl.rc('text', color=fgcolor)
mpl.rc('axes', labelcolor=fgcolor) # edgecolor=fgcolor
mpl.rc('xtick', color=fgcolor)
mpl.rc('ytick', color=fgcolor)
# Global figure that is reused?  Maybe this is faster.
# fig = mpl.figure.Figure(figsize=(8,6), frameon=False, constrained_layout=True)
# canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)

# Plot last event's spectrogram
def plotspectrogram_beat(which=None, figsize=(800,600)):
    event_phase = tg.getfirstevent(which)
    if event_phase is None: return
    event, phase = event_phase

    Fs = tg.getsr()

    # Average pulse lengths.  Don't think individual pulses are measured
    ticp, tocp = tg.getpulses()

    nfft = 1024	# N freq points (FFT size), really nfft/2+1 bins
    shift = 6	# step size
    Np = 257	# window size
    K = 8	# number of tapers
    tm = 6.0	# time support of tapers
    flock = 0.02# frequency locking parameter
    tlock = 8	# time locking parameter

    s, bo = tg.getbeataudio(event)
    if s is None: return

    f = libtfr.fgrid(Fs, nfft)[0]
    S = libtfr.tfr_spec(s, nfft, shift, Np, K, tm, flock, tlock)
    beatms = len(s)/Fs * 1000
    boms = -bo/Fs * 1000
    halfwindow = Np/2/Fs*1000 # Not sure if the bounds are exactly right.  Maybe shift?
    t = np.linspace(boms + halfwindow, boms + beatms - halfwindow, np.shape(S)[1])

    # tfr_spec returns lots of zeros that mess up log10
    S[S==0] = S[S!=0].min()

    dpi = mpl.rcParams['figure.dpi']
    figsize = tuple(s / dpi for s in figsize)
    fig = mpl.figure.Figure(figsize=figsize, frameon=False, constrained_layout=True)
    canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.minorticks_on()
    # Should I divide by Fs here or was that already done?
    mesh = ax.pcolormesh(t, f, 10*np.log10(S), shading='gouraud')

    ax.set_title('Time and Frequency Reassigned Spectrogram ' + ('(Tic)' if phase else '(Toc)'))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(mesh)

    ax.axvline(0, ls=':', c='w', alpha=0.8)
    if phase:
        ax.axvline(-ticp/Fs*1000, ls=':', c='r', alpha=0.7)
    else:
        ax.axvline(-tocp/Fs*1000, ls=':', c='r', alpha=0.7)

    canvas.draw()
    return canvas.buffer_rgba()

# Plot spectrogram of last X seconds
def plotspectrogram_time(length, figsize=(800,600)):
    Fs = tg.getsr()
    # Skip most recent audio to avoid mouse click sound
    skip = int(0.250 * Fs)

    nfft = 1024	# N freq points (FFT size), really nfft/2+1 bins
    shift = 16	# step size
    Np = 257	# window size
    K = 8	# number of tapers
    tm = 6.0	# time support of tapers
    flock = 0.01# frequency locking parameter
    tlock = 8	# time locking parameter

    s, timestamp = tg.getlastaudio(length * Fs + skip)
    if s is None: return
    s = s[:-skip]

    # Hack for time resolution vs speed.  Would be better to take into account
    # how many pixels there are.
    while len(s) / shift > 600 and shift < Np:
        shift = shift * 2

    f = libtfr.fgrid(Fs, nfft)[0]
    S = libtfr.tfr_spec(s, nfft, shift, Np, K, tm, flock, tlock)
    lenms = len(s)/Fs * 1000
    halfwindow = Np/2/Fs*1000 # Not sure if the bounds are exactly right.  Maybe shift?
    t = np.linspace(0 + halfwindow, lenms - halfwindow, np.shape(S)[1])

    # tfr_spec returns lots of zeros that mess up log10
    S[S==0] = S[S!=0].min()

    dpi = mpl.rcParams['figure.dpi']
    figsize = tuple(s / dpi for s in figsize)
    fig = mpl.figure.Figure(figsize=figsize, frameon=False, constrained_layout=True)
    canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.minorticks_on()
    # Should I divide by Fs here or was that already done?
    # Perhaps resolution is high enough to imshow, which should be much faster
    mesh = ax.pcolormesh(t, f, 10*np.log10(S), vmin=-70, vmax=-10, shading='gouraud')

    ax.set_title('Time and Frequency Reassigned Spectrogram')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(mesh)

    events = tg.getevents()
    if events is not None:
        eventms = [(e - timestamp)/Fs * 1000 for e in events]
        eventms = [e for e in eventms if e > t[0] and e < t[-1]]
        for e in eventms: ax.axvline(e, ls=(0, (2, 6)), c='w', alpha=0.25)
        ax2 = ax.secondary_xaxis('top')
        ax2.xaxis.tick_top() # Doesn't seem to work
        ax2.tick_params(axis='x', direction='inout', width=2, length=8)
        ax2.set_xticks(eventms)

    canvas.draw()
    return canvas.buffer_rgba()
