project(
  'clsph',
  'c',
  license: 'MIT',
  default_options: [
    'c_std=c11',
    'optimization=3',
    'buildtype=release',
    'warning_level=3',
    'b_lto=true',
    'b_lto_threads=-1',
    'b_pie=true',
    'prefix=/usr'
  ],
)

debug = get_option('buildtype').startswith('debug')

if debug
  add_project_arguments('-DDEBUG', language : 'c')
endif

data_location = join_paths(
  get_option('prefix'),
  get_option('datadir'),
  'clsph'
)

is_win = host_machine.system() == 'windows'

if is_win
  data_location = '.'
endif

shader_location = join_paths(
  debug ? '' : data_location,
  'shaders',
  ''
)

add_project_arguments(
  [
    '-pedantic',
    #'-Wconversion',
    '-Wshadow',
    '-Wno-unused-parameter',
    '-D_POSIX_C_SOURCE=200809L',
    '-D_GNU_SOURCE',
    '-DSHADER_PATH="@0@"'.format(shader_location),
    '-march=native',
    '-mtune=native',
  ],
  language: 'c'
)

sources = files(
  'src/bphys.c',
  'src/cl.c',
  'src/fluid_params.c',
  'src/log.c',
  'src/sdl.c',
  'src/sph.c',
)

cc = meson.get_compiler('c')

libm = cc.find_library('m', required: false)
sdl = dependency('sdl2')
cl = dependency('OpenCL')
gl = dependency('gl')
epoxy = dependency('epoxy')
thread = dependency('threads')

executable(
  'clsph',
  sources,
  dependencies: [libm, sdl, cl, gl, epoxy, thread],
  install: true,
  #link_args: ['-Wl,--emit-relocs']
  #link_args: ['-fprofile-instr-use']
  #link_args: ['-fprofile-instr-generate']
)

install_subdir(
  'shaders',
  install_dir: data_location
)
