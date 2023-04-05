define([tg_version], m4_normalize(m4_include([version])))
AC_CONFIG_MACRO_DIRS([m4])

AC_INIT([Tg], [tg_version], [vacaboja@gmail.com], [tg-timer], [https://github.com/vacaboja/tg])
AM_INIT_AUTOMAKE([-Wall foreign subdir-objects])

AC_PROG_CC
AC_LANG(C)
AC_CHECK_LIB([pthread], [pthread_mutex_init], [], [AC_MSG_ERROR([pthread not found])])
AC_CHECK_LIB([m], [sqrt], [], [AC_MSG_ERROR([libm not found])])
PKG_CHECK_MODULES([GTK], [gtk+-3.0 glib-2.0])
PKG_CHECK_MODULES([PORTAUDIO], [portaudio-2.0])
PKG_CHECK_MODULES([FFTW], [fftw3f])

AC_DEFUN([AX_PYTHON_PACKAGE],[
  AC_MSG_CHECKING([for Python package $1])
  ac_pyimport_result=`$PYTHON -c "import $1" 2>&1`
  if test $? -eq 0; then
    AC_MSG_RESULT([yes])
    have_$1=yes
  else
    AC_MSG_RESULT([no])
    have_$1=no
    AC_MSG_WARN([cannot import Python module "$1".])
  fi
])

AC_ARG_WITH([python],
    AS_HELP_STRING([--with-python], [Support running Python plugins inside tg]),
    [],
    [with_python=check])
AS_IF([test "x$with_python" != xno], [
  AM_PATH_PYTHON([3.1], [with_python=yes], [
    if test "x$with_python" = xyes; then
      AC_MSG_ERROR([Python not found and --with-python=yes specified])
    fi
    with_python=no])
])
AS_IF([test "x$with_python" != xno], [
  AX_PYTHON_DEVEL([>= '3.1'])

  AC_MSG_CHECKING([if Python building works])
  saved_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $PYTHON_CPPFLAGS"
  AC_COMPILE_IFELSE(
	  [AC_LANG_PROGRAM([#include <Python.h>], [PyModule_Create(NULL);])],
	  [AC_MSG_RESULT([yes])], [AC_MSG_RESULT([no])]
  )

  AX_PYTHON_PACKAGE([numpy])
  if test "x$have_numpy" != xyes; then
    AC_MSG_ERROR([Python support requires numpy])
  fi

  AC_MSG_CHECKING([for Numpy header path])
  numpy_include_path=`$PYTHON -c "import numpy ; print(numpy.get_include())"`
  if test $? -eq 0; then
    AC_MSG_RESULT([$numpy_include_path])
  else
    AC_MSG_ERROR([failed])
  fi
  PYTHON_CPPFLAGS="$PYTHON_CPPFLAGS -I$numpy_include_path"

  CPPFLAGS="$CPPFLAGS $PYTHON_CPPFLAGS -DNPY_NO_DEPRECATED_API"
  AC_CHECK_HEADERS([numpy/ndarrayobject.h], [],
    AC_MSG_ERROR([Couldn't find Numpy headers.])
  )
  CPPFLAGS="$saved_CPPFLAGS"

  AX_PYTHON_PACKAGE([scipy])
  AX_PYTHON_PACKAGE([matplotlib])
  AS_IF([test "x$have_scipy" = xyes -a "x$have_matplotlib" = xyes],
    [AC_DEFINE([HAVE_FILTERGRAPH]) enable_filter_graph=yes;], [enable_filter_graph=no;])
])
AM_CONDITIONAL([PYTHON], [test x$with_python = xyes])

AC_CHECK_TOOL([WINDRES], [windres])
AM_CONDITIONAL([HAVE_WINDRES], [test x$WINDRES != x])

AM_CONDITIONAL([BE_WINDOWS], [test x$OS = xWindows_NT])
AM_COND_IF([BE_WINDOWS], [AC_CONFIG_LINKS([icons/tg-timer.ico:icons/tg-timer.ico])])

CC_CHECK_LDFLAGS([-Wl,--as-needed], [AC_SUBST([AM_LDFLAGS], [-Wl,--as-needed])], [])
CC_CHECK_FLAGS_APPEND([with_cflags], [CFLAGS], [-Wall -Wextra])
AC_SUBST([WARNINGFLAGS], $with_cflags)

AC_CONFIG_FILES([Makefile icons/Makefile])
AC_CONFIG_FILES([packaging/tg-timer.spec:packaging/tg-timer.inc.in:packaging/tg-timer.in.spec])
AC_OUTPUT

AC_MSG_RESULT([
        $PACKAGE_NAME $VERSION
        =====

        prefix:                 ${prefix}

        compiler:               ${CC}
        cflags:                 ${CFLAGS} ${AM_CFLAGS}
        ldflags:                ${LDFLAGS} ${AM_LDFLAGS}

	Python support:         ${with_python}
          Filter plot:          ${enable_filter_graph}
])
