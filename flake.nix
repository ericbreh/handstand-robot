{
  description = "Python Dev Flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
    in {
      devShells.default = with pkgs;
        mkShell {
          buildInputs = [
            python3
            uv
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              stdenv.cc.cc.lib
              libz
              libGL
              glib
              wayland
              libxkbcommon
              xorg.libX11
            ]}:$LD_LIBRARY_PATH
            if [ ! -d ".venv" ]; then
              uv venv
            fi
            source .venv/bin/activate
          '';
        };
    });
}
