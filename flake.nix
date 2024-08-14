{
  description = "vstat dev";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    foolnotion.url = "github:foolnotion/nur-pkg";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs@{ self, flake-parts, nixpkgs, foolnotion }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ foolnotion.overlay ];
          };
          stdenv = pkgs.llvmPackages_18.stdenv;
          pythonPkgs = pkgs.python3Packages.override {
            overrides = self: super: {
              nanobind = super.nanobind.overridePythonAttrs (old: {
                doCheck = false;
                build-system = old.build-system ++ [ pythonPkgs.typing-extensions ];
              });
            };
          };
          nanobind = pythonPkgs.nanobind;
        in
        rec {
          devShells.default = stdenv.mkDerivation {
            name = "vstat-dev";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [
              cmake
              clang_18
              clang-tools_18
              cppcheck
              gdb
              doxygen
              valgrind
              nanobind
            ];
            buildInputs = packages.default.buildInputs
              ++ (with pkgs; [ boost doctest gsl linasm pkg-config ]);
          };

          packages.vstat = stdenv.mkDerivation {
            name = "vstat";
            src = self;
            nativeBuildInputs = with pkgs; [ cmake ];
            buildInputs = with pkgs; [ eve ];

            cmakeFlags = [
              "-DCMAKE_CXX_FLAGS=${
                if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64" else ""
              }"
              "-DCPM_USE_LOCAL_PACKAGES=1"
            ];
          };

          packages.vstat-python = stdenv.mkDerivation {
            name = "vstat";
            src = self;
            nativeBuildInputs = with pkgs; [ cmake ];
            buildInputs = packages.vstat.buildInputs ++ [ nanobind ];
            cmakeFlags = packages.vstat.cmakeFlags ++ [ "-Dvstat_BUILD_PYTHON=ON" ];
          };

          packages.default = packages.vstat;
        };
    };
}
