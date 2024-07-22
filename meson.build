# ----- SPECIFY PROJECT
# eigen dependency requires C++14
project('openbtmixing', 'cpp', default_options: ['cpp_std=c++14'])

# dependencies
# TODO: Can we just clone eigen automatically?
mpi_dep   = dependency('mpi', language : 'cpp')
eigen_dep = dependency('eigen', required : false)
if not eigen_dep.found()
    message('Install eigen and set CPATH to its installation path')
endif
deps = [mpi_dep, eigen_dep]

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
lib = library('openbtmixing', sources : lib_srcs, dependencies : deps)

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
    executable(each[0], each[1], dependencies : deps, link_with : lib)
endforeach

# ----- BUILD UNIT TESTS
# TODO: Integrate these as tests that are run as part of the build
# TODO: amxbrt has build failures
tests_all  = ['crn', 'tree']
tests_all += ['brt', 'brtvp']
#tests_all += ['mbrt', 'ambrt', 'mxbrt', 'amxbrt']
tests_all += ['mbrt', 'ambrt', 'mxbrt']
tests_all += ['sbrt', 'psbrt']
tests_all += ['singlepoisson', 'singlebinomial']
foreach name : tests_all
    test_name = f'test_@name@' 
    test_src = f'src/@test_name@.cpp'
    executable(test_name, test_src, dependencies : deps, link_with : lib)
endforeach
