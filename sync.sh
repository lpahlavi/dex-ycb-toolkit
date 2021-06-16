#!/usr/bin/env bash

###############################################################################
# Display help message
###############################################################################
function display_help() {
    echo "usage: $0 SERVER [options]"
    echo ""
    echo "Required arguments:"
    echo "    SERVER  Identifier for the destination server to synchronize to. Can be"
    echo "            one of the following:"
    echo "              * server3 | ait-server-03"
    echo "              * leonhard"
    echo "              * cnb-workstation"
    echo "              * std-workstation"
    echo ""
    echo "Options:"
    echo "    -h --help         Show this help message."
    echo "    -s --source       Path to the source directory to synchronize to server."
    echo "    -d --destination  Path to the destination directory on the server."

    exit 1
}

###############################################################################
# Parse server name
###############################################################################
case $1 in
    # Leonhard cluster
    server3|ait-server-03)
        SERVER="ait-server-03"
        shift
        ;;
    # AIT student server
    leonhard)
        SERVER="login.leonhard.ethz.ch"
        shift
        ;;
    # Workstation in CNB basement
    cnb-workstation)
        SERVER="cnb-d102-44.inf.ethz.ch"
        shift
        ;;
    # Workstation at AIT Lab in STD
    std-workstation)
        SERVER="129.132.75.179"
        shift
        ;;
    # Unknown server
    *)
        echo "[ERROR] Unknown server name '${1}'"
        display_help
        exit 0
        ;;
esac

###############################################################################
# Parse options
###############################################################################
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    # Display help
    -h|--help)
        display_help
        exit 0
        ;;
    # Source directory
    -s|--source)
        SOURCE_DIRECTORY=$2
        shift
        shift
        ;;
    # Destination directory
    -d|--destination)
        DESTINATION_DIRECTORY=$2
        shift
        shift
        ;;
    # Any other value is unsupported
    *)
        echo "[ERROR] Unknown option '${key}'"
        display_help
        exit 0
esac
done

###############################################################################
# Synchronize command
###############################################################################
SOURCE_DIRECTORY="${SOURCE_DIRECTORY:-.}"
DESTINATION_DIRECTORY="${DESTINATION_DIRECTORY:-master-thesis/dex-ycb-toolkit}"

rsync -a --progress \
    --exclude .git \
    --exclude-from .gitignore \
    --exclude-from .git/info/exclude \
    $SOURCE_DIRECTORY pahlavil@$SERVER:$DESTINATION_DIRECTORY
