{
  description = "vstat dev";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";

  outputs = { self, flake-utils, nixpkgs, nur }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nur.overlay ];
          };
          buildTesting = pkgs.targetPlatform.isx86_64;
        in rec
        {
          defaultPackage = pkgs.gcc12Stdenv.mkDerivation {
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
                pkgs.nur.repos.foolnotion.cmake-init
                pkgs.nur.repos.foolnotion.eve
                pkgs.nur.repos.foolnotion.vectorclass
              ] ++ lib.optionals buildTesting [ pkgs.nur.repos.foolnotion.linasm ];
          };

          devShell = pkgs.gcc12Stdenv.mkDerivation {
            name = "vstat-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ cmake clang_14 clang-tools cppcheck ];

            buildInputs = defaultPackage.buildInputs;

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc12Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
