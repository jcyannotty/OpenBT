# ----- SPECIFY PROJECT
# eigen dependency requires C++14
project('openbtmixing', 'cpp', default_options: ['cpp_std=c++14'])

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

if get_option('mpi') != 'none'
    # TODO: This should probably set -DUSE_MPI since the code is not written
    # explicitly for OpenMPI.
    add_project_arguments('-D_OPENMPI', language: 'cpp')
endif

if get_option('mpi') == 'open-mpi'
    deps_all += [dependency('mpi', language : 'cpp')]
elif get_option('mpi') == 'mpich'
    # Currently meson does not find and work with MPICH as expected if we just
    # declare MPI as a dependence.  This assumes that MPICH can be discovered
    # through pkg-config.
    # TODO: Add link to MPI issue
    # TODO: Add link to project that I based this on SU(2)?
    deps_all += [dependency('mpich')]
endif

# ----- SPECIFY & BUILD libopenbtmixing LIBRARY
# Sources to build individual components in single library
# TODO: How to get rid of src/?
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
# TODO: Put headers in their own includes folder?
# TODO: Way to get these automatically without explicit listing?
lib_headers = include_directories('src')
install_headers('src/ambrt.h',
                'src/amxbrt.h',
                'src/brt.h',
                'src/brtfuns.h',
                'src/crn.h',
                'src/dinfo.h',
                'src/mbrt.h',
                'src/mxbrt.h',
                'src/psbrt.h',
                'src/rn.h',
                'src/sbrt.h',
                'src/singlebinomial.h',
                'src/singlepoisson.h',
                'src/tnorm.h',
                'src/tree.h',
                'src/treefuns.h')

lib_srcs  = src_crn + src_tree + src_brt + src_mbrt + src_sbrt
lib_srcs += src_mxbrt + src_ambrt + src_psbrt + src_amxbrt
lib_srcs += src_poisson + src_binomial
lib = library('openbtmixing', sources : lib_srcs,
              dependencies : deps_all,
              include_directories : lib_headers,
              pic : true, install : true)

# ----- BUILD COMMAND LINE TOOLS
CLTs = [['openbtcli',        'src/cli.cpp'],
        ['openbtpred',       'src/pred.cpp'],
        ['openbtvartivity',  'src/vartivity.cpp'],
        ['openbtsobol',      'src/sobol.cpp'],
        ['openbtmopareto',   'src/mopareto.cpp'],
        ['openbtmixingwts',  'src/mixingwts.cpp'],
        ['openbtmixing',     'src/mixandemulate.cpp'],
        ['openbtmixingpred', 'src/mixandemulatepred.cpp']]
foreach each : CLTs
    executable(each[0], each[1],
               dependencies : deps_all, link_with : lib,
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
    test_src = f'src/@test_name@.cpp'
    executable(test_name, test_src,
               dependencies : deps_all, link_with : lib,
               install : true)
endforeach
