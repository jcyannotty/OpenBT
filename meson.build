# ----- SPECIFY PROJECT
# eigen dependency requires C++14
project('openbtmixing', 'cpp',
        default_options: ['cpp_std=c++14'],
        meson_version: '>=1.6.0')

# ----- DEPENDENCIES
# If users have a local install of eigen that can be discovered with
# pkg-config, then that installation will be used.  Otherwise, meson will
# download the specific tarball as defined in eigen.wrap and use that.
eigen_dep = dependency('eigen', fallback : ['eigen', 'eigen_dep'])
deps_all = [eigen_dep]

# ----- USER ARGUMENTS
# Run this after dependencies so that we can alter dependencies according to
# the user's needs.
if not get_option('verbose')
    add_project_arguments('-DSILENT', language: 'cpp')
endif

if get_option('use_mpi')
    add_project_arguments('-D_OPENMPI', language: 'cpp')
    # This works with both OpenMPI & MPICH beginning with v1.6.0.  See
    # https://github.com/mesonbuild/meson/pull/13619
    deps_all += [dependency('mpi', language : 'cpp')]
endif

# ----- SPECIFY HEADERS USED TO BUILD LIBRARY, CLTs, & TESTS
# Install the headers along with library for linking
install_headers('includes/ambrt.h',
                'includes/amxbrt.h',
                'includes/brt.h',
                'includes/brtfuns.h',
                'includes/crn.h',
                'includes/dinfo.h',
                'includes/mbrt.h',
                'includes/mxbrt.h',
                'includes/psbrt.h',
                'includes/rn.h',
                'includes/sbrt.h',
                'includes/singlebinomial.h',
                'includes/singlepoisson.h',
                'includes/tnorm.h',
                'includes/tree.h',
                'includes/treefuns.h')
incdir = include_directories('includes')

# ----- SPECIFY & BUILD libopenbtmixing LIBRARY
# Sources to build individual components in single library
src_crn      = ['src/crn.cpp', 'src/tnorm.cpp']
src_tree     = ['src/treefuns.cpp', 'src/tree.cpp']
src_brt      = ['src/brt.cpp', 'src/brtmoves.cpp', 'src/brtfuns.cpp']
src_mbrt     = ['src/mbrt.cpp']
src_sbrt     = ['src/sbrt.cpp']
src_mxbrt    = ['src/mxbrt.cpp']
src_ambrt    = ['src/ambrt.cpp']
src_psbrt    = ['src/psbrt.cpp']
src_amxbrt   = ['src/amxbrt.cpp']
src_poisson  = ['src/singlepoisson.cpp']
src_binomial = ['src/singlebinomial.cpp']

# Build library
lib_srcs  = src_crn + src_tree + src_brt + src_mbrt + src_sbrt
lib_srcs += src_mxbrt + src_ambrt + src_psbrt + src_amxbrt
lib_srcs += src_poisson + src_binomial
lib = library('openbtmixing', sources : lib_srcs,
              dependencies : deps_all,
              include_directories : incdir,
              pic : true, install : true)

# ----- BUILD COMMAND LINE TOOLS
# Full set of OpenBT command line tools built as standalone binaries rather than
# linked to openbtmixing library.
CLTs = [['openbtcli',        'src/cli.cpp'],
        ['openbtpred',       'src/pred.cpp'],
        ['openbtvartivity',  'src/vartivity.cpp'],
        ['openbtsobol',      'src/sobol.cpp'],
        ['openbtmopareto',   'src/mopareto.cpp'],
        ['openbtmixingwts',  'src/mixingwts.cpp'],
        ['openbtmixing',     'src/mixandemulate.cpp'],
        ['openbtmixingpred', 'src/mixandemulatepred.cpp']]
foreach each : CLTs
    executable(each[0], [each[1]] + lib_srcs,
               dependencies : deps_all,
               include_directories : incdir,
               install : true)
endforeach

# ----- BUILD UNIT TESTS
# TODO: amxbrt has build failures
# TODO: Some of these tests are failing with segfaults.
# TODO: Integrate these as tests that are run as part of the build once they
# are all passing.
tests_all  = ['crn', 'tree']
tests_all += ['brt', 'brtvp']
#tests_all += ['mbrt', 'ambrt', 'mxbrt', 'amxbrt']
tests_all += ['mbrt', 'ambrt', 'mxbrt']
tests_all += ['sbrt', 'psbrt']
tests_all += ['singlepoisson', 'singlebinomial']
foreach name : tests_all
    test_name = f'test_@name@' 
    test_src = f'tests/test_@name@.cpp'
    executable(test_name, test_src,
               dependencies : deps_all, link_with : lib,
               include_directories : incdir,
               install : true)
endforeach
