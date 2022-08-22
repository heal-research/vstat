{
  description = "vstat dev";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.foolnotion.url = "github:foolnotion/nur-pkg";
  inputs.nixpkgs.url = "github:nixos/nixpkgs";

  outputs = { self, flake-utils, nixpkgs, foolnotion }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ foolnotion.overlay ];
          };
          buildTesting = pkgs.targetPlatform.isx86_64;
        in rec
        {
          packages.default = pkgs.stdenv.mkDerivation {
            name = "vstat";
            src = self;

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              "-DBUILD_TESTING=${if buildTesting then "ON" else "OFF"}"
            ];

            nativeBuildInputs = with pkgs; [ cmake ];
            buildInputs = with pkgs; [
                # python environment for bindings and scripting
                boost
                doctest
                gsl
                pkg-config
                eve
              ] ++ lib.optionals buildTesting [ linasm ];
          };

          devShells.default = pkgs.stdenv.mkDerivation {
            name = "vstat-dev";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ cmake clang_14 clang-tools cppcheck gdb ];

            buildInputs = packages.default.buildInputs;
          };
        }
      );
}
