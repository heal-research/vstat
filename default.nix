let pkgs = import <nixos-unstable> {};
in
pkgs.gcc10Stdenv.mkDerivation {
  name = "vstat-env";
  hardeningDisable = [ "all" ];

  buildInputs = with pkgs; [
        boost
        clang_10
        cmake
        doctest
        gdb
        gsl
        linasm
        linuxPackages.perf
        ninja
        openblas
        pkg-config
        valgrind
      ];
    }

