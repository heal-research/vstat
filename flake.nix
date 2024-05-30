{
  description = "vstat dev";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.foolnotion.url = "github:foolnotion/nur-pkg";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, foolnotion }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };
        stdenv_ = pkgs.llvmPackages_18.stdenv;
        pythonVersion = "${pkgs.python3.sourceVersion.major}.${pkgs.python3.sourceVersion.minor}";
        nanobind = pkgs.python3Packages.nanobind;
      in
      rec {
        devShells.default = stdenv_.mkDerivation {
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

        packages.default = stdenv_.mkDerivation {
          name = "vstat";

          src = self;

          nativeBuildInputs = with pkgs; [ cmake ];

          buildInputs = with pkgs; [
            eve
            nanobind
          ];

          cmakeFlags = [
            "-Dvstat_BUILD_PYTHON=ON"
            "-DCMAKE_CXX_FLAGS=${
              if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64" else ""
            }"
            "-DCMAKE_PREFIX_PATH=${nanobind}/lib/python${pythonVersion}/site-packages"
          ];
        };
      });
}
