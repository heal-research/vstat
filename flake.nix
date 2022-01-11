{
  description = "vstat dev";
  nixConfig.bash-prompt = "\n\\[\\e[93m\\e[1m\\][vstat-dev:\\[\\e[92m\\e[1m\\]\\w]$\\[\\e[0m\\] ";

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
        in
        {
          devShell = pkgs.gcc11Stdenv.mkDerivation {
            name = "vstat-env";
            hardeningDisable = [ "all" ];
            impureUseNativeOptimizations = true;
            nativeBuildInputs = with pkgs; [ cmake clang_13 clang-tools cppcheck ];
            buildInputs = with pkgs; [
                # python environment for bindings and scripting
                boost
                doctest
                gsl
                pkg-config
                pkgs.nur.repos.foolnotion.cmake-init
                pkgs.nur.repos.foolnotion.linasm
                pkgs.nur.repos.foolnotion.vectorclass
              ];

            shellHook = ''
              LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]};
              '';
          };
        }
      );
}
