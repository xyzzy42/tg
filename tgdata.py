import numpy as np
import json

class TgData:
    """Load one or more jSON dumps from tg-timer

    This loads the jSON data from the dumps and presents it in what might be a more
    useful manner.  It can load multiple dump files and will merge them, taking into
    account that one dump might overlap with the next.

    Attributes
    ----------
    Fs : float
        The audio sample rate in Hz.
    la : float
        The lift angle.
    n : int
        The number of beat measurements.  Same as `len(beats)`.
    beats : ndarray of uint64
        The times of each beat, tic or toc, in samples.
        This needs to be divided by the sample rate to get time in seconds.
    tictoc : ndarray of bool
        Is the beat a tic (true) or a toc (false).  Matches the `beats` array.
    beatidx : ndarray of int
        The number of each beat in `beats`.  If the data was perfect, this would simply
        be the array [0, 1, 2, ..., n-1].  But if the data is not perfect and a single
        beat was missed, it can be detected and the array will skip an number, e.g. [0,
        1, 2, 4, 5, ...  ] if beat 3 was not detected.
    amps : ndarray of float
        The amplitude measurements.
    amptimes : ndarray of uint64
        The time assigned to the aplitude meansurements.
        Note that these times are not the as the times in the `beats` attribute.  Also
        note that amplitudes measurements are derived from more data from more than one
        single oscillation.  The time points are the centers of the measurement
        intervals.
    """

    def __init__(self, data):
        """Create object from jSON dump files

        Parameters
        ----------
        data : iterable of str or dict

        data should be a list or other iterable of file names.  The files should be in
        order, oldest dump to newest.

        It's also possible for data to be a single dict with the same contents that
        would be created by loading a jSON file, e.g. with `json.load(filename)`.

        Raises
        ------
        RuntimeError
            The files are not mergable.  Different sampling rates, don't overlap, not
            in order, etc.
        """
        def from_json(name):
            with open(name) as f:
                j = json.load(f)
                j['filename'] = name
                return j

        def merge(name, othername, othertype):
            events = np.array(data[0][name]['times'], dtype=np.uint64)
            other = np.array(data[0][name][othername], dtype=othertype)
            for d in data[1:]:
                new = np.array(d[name]['times'], dtype=np.uint64)
                offset = events.searchsorted(new[0])
                end = events.size - offset
                if not np.array_equal(events[offset:], new[0:end]):
                    raise RuntimeError(f"Dump file {d['filename']} doesn't overlap")
                events = np.append(events, new[end:])
                other = np.append(other, np.array(d[name][othername][end:], dtype=othertype))

            return (events, other)

        if isinstance(data, dict):
            data = [data]
        else:
            if isinstance(data, str):
                data = [data]
            data = [from_json(name) for name in data]


        # Check matching sample rates and beat rates
        self.Fs = data[0]['sample_rate']
        self.bph = data[0]['bph']
        for d in data[1:]:
            if self.Fs != d['sample_rate']:
                raise RuntimeError(f"Dump file {d['filename']} doesn't have a matching sample rate {self.Fs}")
            if self.bph != d['bph']:
                raise RuntimeError(f"Dump file {d['filename']} doesn't have a matching beat rate {self.bph}")

        # Merge beat data
        self.beats, self.tictoc = merge('beats', 'tictoc', np.bool)
        self.n = len(self.beats)

        # Check for identical lift angle
        self.la = data[0]['amplitude']['liftangle']
        for d in data[1:]:
            if self.la != d['amplitude']['liftangle']:
                raise RuntimeError(f"Dump file {d['filename']} doesn't have matching lift angle {self.la}")

        # Merge amplitude data
        self.amptimes, self.amps = merge('amplitude', 'values', np.float32)
        self.amps *= self.la

        # Start time at zero
        self.amptimes -= self.beats[0]
        self.beats -= self.beats[0]

        # Detect single beat skips
        skips, = np.nonzero(~np.diff(self.tictoc))
        delta = np.ones(self.n, dtype=int)
        delta[0] = 0
        delta[skips+1] = 2
        self.beatidx = delta.cumsum();

    def est_rate(self, start=0, stop=None):
        """Estimate rate with linear list squares.

        Parameters
        ----------
        start : int
           Index of data to use for start of estimation.  Default 0.
        length : int
           Index to stop at. Default all data points.

        This uses a least squares fit to estimate the rate.  This is more accurate than
        simply using how far off the watch is at the end of the sampling interval.
        However, it is sensitive to missed beats.

        Returns
        -------
        Rate in samples per beat

        Note the units, it's not bph. `data.Fs / data.est_rate()` would convert to beats / sec.
        Multiply by 3600 to gets beats / hour.
        """
        if stop is None:
            stop = self.n
        return np.linalg.lstsq(
            np.stack([self.beatidx[start:stop], np.ones(stop-start)], 1),
            self.beats[start:stop],
            None)[0][0]

    def est_spd(self, start=0, stop=None):
        """Estimate rate error with linear list squares in sec/day.

        See `est_rate()`.  This is the same, but returns the rate as the error in sec/day.
        """

        return self.rate_to_spd(self.est_rate(start, stop))


    def rate_to_spd(self, rate):
        """Convert a rate into s/d error.

        Parameters
        ----------
        rate : float
            Rate in samples/beat.

        This converts a rate, in samples per beat, into an error in s/d, based on the
        expected bph.

        Returns
        -------
        Error in seconds per day.
        """

        return 24 * (3600 - self.bph * rate / self.Fs)

    def fixup_gaps(self, seglen=330):
        """Attempt to fix missing beats.

        This will look for gaps where multiple beats are missing and adjust the beatidx
        array to account for them.  The initial fix done when the data is loaded looks
        for two tics or two tocs in a row to detect this missing toc or tic between
        them.  This function uses the time of the beats and looks for gaps that are too
        long.  It won't work if the watch is wildly inconsistent or being removed and
        reattached to the microphone.
        """

        exp_m = self.Fs * 3600 / self.bph
        d = np.zeros(self.n, dtype=int)
        d[0] = self.beatidx[0]
        for i in range(1, self.n, seglen):
            stop = min(self.n, i + seglen)
            m = self.est_rate(i - 1, stop)
            if m > exp_m * 1.1 or m < exp_m * 0.9:
                print(f"Rate estimation off by {(m - exp_m) * 100/ exp_m:.0f}% between {i-1} - {stop}")
            np.rint(np.diff(self.beats[i-1:stop]) / m, casting='unsafe', out=d[i:stop])
        d[d == 0] = 1

        d.cumsum(out=self.beatidx)
        return d

