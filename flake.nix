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
      in rec {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "vstat";
          src = self;

          cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];

          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = with pkgs; [ eve ];
        };

        devShells.default = pkgs.stdenv.mkDerivation {
          name = "vstat-dev";
          hardeningDisable = [ "all" ];
          impureUseNativeOptimizations = true;
          nativeBuildInputs = with pkgs; [
            cmake
            clang_15
            clang-tools
            cppcheck
            gdb
          ];
          buildInputs = packages.default.buildInputs
            ++ (with pkgs; [ boost doctest gsl linasm pkg-config ]);
        };
      });
}
