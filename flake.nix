{
  description = "vstat dev";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.foolnotion.url = "github:foolnotion/nur-pkg";
  inputs.nixpkgs.url = "github:nixos/nixpkgs";

  outputs = { self, flake-utils, nixpkgs, foolnotion }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };
        stdenv_ = pkgs.llvmPackages_16.stdenv;
      in rec {
        packages.default = stdenv_.mkDerivation {
          name = "vstat";
          src = self;

          cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];

          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = with pkgs; [ eve ];
        };

        devShells.default = stdenv_.mkDerivation {
          name = "vstat-dev";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
          nativeBuildInputs = with pkgs; [
            cmake
            clang_16
            clang-tools_16
            cppcheck
            gdb
          ];
          buildInputs = packages.default.buildInputs
            ++ (with pkgs; [ boost doctest gsl linasm pkg-config ]);
        };
      });
}
