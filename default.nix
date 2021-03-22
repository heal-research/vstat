let pkgs = import <nixos-unstable> {};
in
pkgs.gcc10Stdenv.mkDerivation {
  name = "vstat-env";
  hardeningDisable = [ "all" ];

  buildInputs = with pkgs; [
        gdb
        valgrind
        linuxPackages.perf
        cmake
        doctest
        clang_10
        ninja
        gsl
        pkg-config
      ];
    }

