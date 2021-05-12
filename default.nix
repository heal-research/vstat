# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/heads/master.zip") { };
in
pkgs.gcc11Stdenv.mkDerivation {
  name = "vstat-env";
  hardeningDisable = [ "all" ];

  buildInputs = with pkgs; [
        boost
        clang_11
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

