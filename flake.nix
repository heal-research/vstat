{
  description = "vstat dev";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      nixpkgs,
      foolnotion,
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ foolnotion.overlay ];
          };
          #stdenv = pkgs.llvmPackages_latest.stdenv;
          inherit (pkgs.llvmPackages_latest) stdenv;
        in
        rec {
          devShells.default = stdenv.mkDerivation {
            name = "vstat-dev";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [
              cmake
              clang-tools
              cppcheck
              gdb
              eve
              doxygen
              valgrind
              pkgs.python3Packages.nanobind
            ];
            buildInputs =
              packages.default.buildInputs
              ++ (with pkgs; [
                boost
                doctest
                gsl
                pkg-config
              ])
              ++ (with pkgs; if pkgs.stdenv.isx86_64 then [ linasm ] else [ ]);
          };

          packages.vstat = stdenv.mkDerivation {
            name = "vstat";
            src = self;
            nativeBuildInputs = with pkgs; [ cmake ];
            buildInputs = with pkgs; [ eve ];

            cmakeFlags = [
              "-DCMAKE_CXX_FLAGS=${if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64" else ""}"
              "-DCPM_USE_LOCAL_PACKAGES=1"
            ];
          };

          packages.vstat-python = stdenv.mkDerivation {
            name = "vstat";
            src = self;
            nativeBuildInputs = with pkgs; [ cmake ];
            buildInputs = packages.vstat.buildInputs ++ [ pkgs.python3Packages.nanobind ];
            cmakeFlags = packages.vstat.cmakeFlags ++ [ "-Dvstat_BUILD_PYTHON=ON" ];
          };

          packages.default = packages.vstat;
        };
    };
}
