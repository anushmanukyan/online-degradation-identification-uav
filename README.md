# Online Degradation Identification of UAV

This command line tool allows to perform an online degradation identification of UAV's hardware-related components. By analyzing analysing flight data stream from a UAV following a pre-defined mission it predicts the level of degradation of these components at early stages.

## How to use

Install the requirements using pip:

    pip install -r requirements.txt


Type this command to see all available commands:

    python droneML.py --help


If there is an issue with importing matplotlib related to the locale, then export the following environment variables:

    export LC_ALL=en_US.UTF-8
    export LANG=en_UTF-8


## Example commands

    python droneML.py online -k 3 -w 200 --window-size 3 --train training/drop/*.csv --test test/*.csv


## Publication

* [UAV degradation identification for pilot notification using machine learning techniques](https://orbilu.uni.lu/handle/10993/32873)
* [Real time degradation identification of UAV using machine learning techniques](https://orbilu.uni.lu/handle/10993/32968)

