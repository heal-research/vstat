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

          buildTests = if "${system}" == "aarch64-linux" then false else true;
        in rec
        {
          defaultPackage = pkgs.gcc11Stdenv.mkDerivation {
            name = "vstat-test";
            src = self;

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              "-DBUILD_TESTING=${if buildTests then "ON" else "OFF"}"
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
              ] ++ lib.optionals buildTests [ pkgs.nur.repos.foolnotion.linasm ];
          };

          devShell = pkgs.gcc11Stdenv.mkDerivation {
            name = "vstat-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ cmake clang_13 clang-tools cppcheck ];

            buildInputs = defaultPackage.buildInputs;

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
