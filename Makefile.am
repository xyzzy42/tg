SUBDIRS = icons
bin_PROGRAMS = tg-timer
EXTRA_PROGRAMS = tg-timer-dbg \
		 tg-timer-prf \
		 tg-timer-vlg
MOSTLYCLEANFILES = $(EXTRA_PROGRAMS)

tg_timer_SOURCES = src/algo.c \
		   src/audio.c \
		   src/computer.c \
		   src/config.c \
		   src/interface.c \
		   src/audio_interface.c \
		   src/output_panel.c \
		   src/serializer.c \
		   src/filter_interface.c \
		   src/tppm.c \
		   src/tg.h 

if PYTHON
tg_timer_SOURCES += src/python.c
nodist_tg_timer_SOURCES = \
	src/py/spectrogram.c \
	src/py/filter_graph.c
CLEANFILES = $(nodist_tg_timer_SOURCES)
endif

tg_timer_dbg_SOURCES = $(tg_timer_SOURCES)
tg_timer_prf_SOURCES = $(tg_timer_SOURCES)
tg_timer_vlg_SOURCES = $(tg_timer_SOURCES)
nodist_tg_timer_dbg_SOURCES = $(nodist_tg_timer_SOURCES)
nodist_tg_timer_prf_SOURCES = $(nodist_tg_timer_SOURCES)
nodist_tg_timer_vlg_SOURCES = $(nodist_tg_timer_SOURCES)

LIBS = $(GTK_LIBS) \
       $(GTHREAD_LIBS) \
       $(PORTAUDIO_LIBS) \
       $(FFTW_LIBS) \
       $(PYTHON_LIBS) $(PYTHON_EXTRA_LIBS) \
       -lpthread \
       -lm

AM_CPPFLAGS = -DPROGRAM_NAME=\"Tg\" -DVERSION=\"$(PACKAGE_VERSION)\"
tg_timer_dbg_CPPFLAGS = $(AM_CPPFLAGS) -DDEBUG

AM_CFLAGS = $(WARNINGFLAGS) \
	    $(GTK_CFLAGS) \
	    $(GTHREAD_CFLAGS) \
	    $(PORTAUDIO_CFLAGS) \
	    $(FFTW_CFLAGS) \
	    $(PYTHON_CPPFLAGS) $(PYTHON_EXTRA_CFLAGS)
tg_timer_dbg_CFLAGS = $(AM_CFLAGS) -ggdb
tg_timer_prf_CFLAGS = $(AM_CFLAGS) -pg
tg_timer_vlg_CFLAGS = $(AM_CFLAGS) -g

tg_timer_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_EXTRA_LDFLAGS)
tg_timer_dbg_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_EXTRA_LDFLAGS)
tg_timer_prf_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_EXTRA_LDFLAGS)
tg_timer_vlg_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_EXTRA_LDFLAGS)

if BE_WINDOWS
if HAVE_WINDRES
tg_timer_SOURCES += icons/tg-timer.rc
endif
tg_timer_LDFLAGS += -mwindows
tg_timer_dbg_LDFLAGS += -mconsole
tg_timer_prf_LDFLAGS += -mconsole
tg_timer_vlg_LDFLAGS += -mconsole
endif

desktopdir = $(datadir)/applications
dist_desktop_DATA = icons/tg-timer.desktop

mimedir = $(datadir)/mime/packages
dist_mime_DATA = icons/tg-timer.xml

dist_man_MANS = docs/tg-timer.1

%.c : %.py Makefile
	echo 'const char $(*F)_py[] =' > $@
	sed -e 's/\\/\\\\/g;s/"/\\"/g;s/  /\\t/g;s/^/"/;s/$$/\\n"/' $< >> $@
	echo ";" >> $@

EXTRA_DIST = autogen.sh \
	     icons \
	     packaging \
	     LICENSE \
	     README.md

.rc.o:
	$(WINDRES) $< -O coff -o $@

test: tg-timer-dbg
	./tg-timer-dbg test
.PHONY: test

valgrind: tg-timer-vlg
	valgrind --leak-check=full -v --num-callers=99 --suppressions=.valgrind.supp ./$^
.PHONY: valgrind
