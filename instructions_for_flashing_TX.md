FLASHING JETSON TX2 STEPS:

Install WSL ubuntu 16.04
set to wsl 2
download the sdk manager file sdkmanager_2.2.0-12028_amd64.deb from nvidia website

Put jetson in forced recovery mode:
turn off and on
hold REC button (one left of power button)
press and release RST button (far left)
keep holding REC button for two seconds

IN POWERSHELL:
winget install --interactive --exact dorssel.usbipd-win

RESTART COMPUTER

ENTER WSL:
cd ../

cd ../

cd /mnt/c/Users/<your user>/<folder that .deb file is in>

sudo apt update 

sudo apt install -y gdebi-core

sudo gdebi sdkmanager_2.2.0-12028_amd64.deb

sudo apt install iputils-ping iproute2 netcat iptables dnsutils network-manager usbutils net-tools python3-yaml dosfstools libgetopt-complete-perl openssh-client binutils vim-common cpio udev dmidecode -y

sudo apt install linux-tools-virtual hwdata

EXIT WSL:

IN POWERSHELL:

usbipd.exe list

Identify the BUS ID of the Jetson (0955:7020 means its not in recovery mode, 0955:7c18 means its in recovery mode)
the bus id should looks something like 1-2
Attach the BUS ID to the WSL Linux distribution by running the following command:

usbipd.exe bind --busid <BUSID> --force

usbipd.exe attach --wsl --busid=<BUSID> --auto-attach

OPEN A NEW POWERSHELL TAB TO USE WSL, DO NOT STOP THE INFINITE ATTACH LOOP

ENTER WSL:
Validate that the Jetson device appears in the WSL Linux distribution by running the following command:

lsusb

sdkmanager --cli
Log in, make nvidia developer account

in menus select these options:
┏ Install options ------------------------------ ┓
- Select action: Install
- Select product: Jetson
- Select system configuration: Host Machine [Ubuntu 16.04 - x86_64 WSL], Target Hardware
- Select target hardware: Jetson TX2 modules
- Select target operating system: Linux
- Do you want to show all release versions? Yes
- Select SDK version: JetPack 4.6.5
- Select additional SDKs: None
- Do you want to customize the install settings? Yes
- Do you want to flash Jetson TX2 module? Yes
- Download folder: /home/lrfield/Downloads/nvidia/sdkm_downloads
- Target HW image folder: /home/lrfield/nvidia/nvidia_sdk
- Accept the terms and conditions of all license agreements: I accept the terms and conditions of all license agreements
┗ ---------------------------------------------- ┛

It will prompt you to run this command, then it will open a menu where you put in your sudo password and it installs everything (when it asks you to flash the device select yes):
sdkmanager --cli --action install --login-type devzone --product Jetson --target-os Linux --version 4.6.5 --show-all-versions --host --target JETSON_TX2_TARGETS --flash --license accept
