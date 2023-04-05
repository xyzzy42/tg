# A program for timing mechanical watches

The program tg is distributed under the GNU GPL license version 2. The full
source code of tg is available at
[https://github.com/xyzzy42/tg](https://github.com/xyzzy42/tg) and its
copyright belongs to the respective contributors.

Tg is in development, and there is still no manual. Some info can be found
in this
[thread at WUS](http://forums.watchuseek.com/f6/open-source-timing-software-2542874.html),
in particular the calibration procedure is described at
[this post](http://forums.watchuseek.com/f6/open-source-timing-software-2542874-post29970370.html).

## Install instructions

Tg is known to work under Microsoft Windows, OS X, and Linux. Moreover it
should be possible to compile the source code under most modern UNIX-like
systems. See the sub-sections below for the details.

### Windows

Binaries can be found at https://tg.ciovil.li

Unfortunately, these packages have not been updated since 2017.  Help from
someone who can build the Windows installer version would be appreciated. 
You'll need to [build from source](#compiling-on-windows) to get any features
from the last five plus years.

### Macintosh

A formula for the Homebrew package manager has been prepared by GitHub user
[dmnc](https://github.com/dmnc) and then updated by
[xyzzy42](https://github.com/xyzzy42).  To use it, you need to install Homebrew
first (instructions on http://brew.sh).

Then run the following command to check everything is set up correctly
and follow any instructions it gives you.

	brew doctor

To install tg, run:

	brew install --HEAD xyzzy42/horology/tg-timer
	
You can now launch tg by typing:

	tg-timer &

### Debian or Debian-based (e.g. Mint, Ubuntu)

Binary .deb packages can be downloaded from https://tg.ciovil.li

Unfortunately, these packages have not been updated since 2017.

### Fedora, CentOS or other Redhat-based

Binary RPM packages are available from https://copr.fedorainfracloud.org/coprs/tpiepho/tg-timer/

This COPR repository can be added to dnf's list with:
```sh
dnf copr enable tpiepho/tg-timer
```
Then tg-timer can be installed with `dnf install tg-timer`, or with any dnf
based GUI package installer.

## Compiling from sources

The source code of tg can probably be built by any C99 compiler, however only
gcc and clang have been tested.  You need the following libraries:  gtk+3,
portaudio2, fftw3 (all available as open source).

The optional plotting features for audio filter response graphs and audio
spectrograms require Python.  The follow Python modules are needed for all
features:  numpy, scipy, matplotlib, libtfr.

### Normal build
Get source:
```sh
git clone https://github.com/xyzzy42/tg.git
cd tg

# Optional, check out a specific branch other than the default
git checkout new-stuff
```

Build it:
```sh
./autogen.sh
./configure
make
```

### Debug build
After the steps of a normal build, above, run:
```sh
make tg-timer-dbg
```

### Compiling on Windows

It is suggested to use the msys2 platform. First install msys2 according
to the instructions at [http://www.msys2.org](http://www.msys2.org). Then
issue the following commands to install dependencies:

```sh
pacman -S mingw-w64-x86_64-gcc make pkg-config mingw-w64-x86_64-gtk3 mingw-w64-x86_64-portaudio mingw-w64-x86_64-fftw git autoconf automake libtool
```

Then follow the [normal build](#normal-build) instructions to clone the
repository and build it.

### Compiling on Debian

To compile tg on Debian, install these dependencies:

```sh
sudo apt-get install libgtk-3-dev libjack-jackd2-dev portaudio19-dev libfftw3-dev git autoconf automake libtool
```

Additional software is necessary for the optional Python graphing system, see
Fedora section for more information.  Debian specific instructions welcome.

Then follow the [normal build](#normal-build) instructions to clone the
repository and build it.

The package libjack-jackd2-dev is not necessary, it only works around a
known bug (https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=718221).

### Compiling on Fedora

To compile tg on Fedora, install these dependencies:

```sh
sudo dnf install fftw-devel portaudio-devel gtk3-devel autoconf automake libtool
```

To use the Python graphing code (filter response and audio spectrogram) one
should also install:
```sh
sudo dnf install python3-devel python3-numpy python3-scipy python3-matplotlib
pip install --user libtfr
```

Then follow the [normal build](#normal-build) instructions to clone the
repository and build it.

To build an RPM on Fedora or another RPM based distro, install the build
prerequisites and checkout the source as for compiling (above), then run
`rpmbuild` to create the RPM:

```sh
rpmbuild --build-in-place -bb packaging/tg-timer.spec
```


### Compiling on Macintosh

To build on MacOS, install the follow dependencies with
[Homebrew](http://brew.sh).  If you do not have Homebrew, the link above will
explain the install.

```sh
brew install pkg-config autoconf automake libtool gtk+3 portaudio fftw gnome-icon-theme
```

To use the Python graphing code (filter response and audio spectrogram) one
should also install:

```sh
brew install python numpy scipy
pip3 install matplotlib libtfr
```

If you have multiple versions of python3 installed, you might need to use a
specific version of pip, e.g. `pip3.11` in the above command.

Then follow the [normal build](#normal-build) instructions to clone the
repository and build it.

If you have multiple versions of Python installed and the configure script does
not detect the correct one, i.e. the one for which numpy, matplotlib, libtfr,
etc. have been installed for, then run the configure script as:

```sh
PYTHON=python3.11 ./configure
```

Where `python3.11` is the correct version to use.
