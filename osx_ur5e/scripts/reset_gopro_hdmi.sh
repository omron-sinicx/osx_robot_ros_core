#!/bin/bash
# Reset the UGREEN HDMI capture card USB device to recover from UVC URB errors.
# Run this when /dev/video0 stops working after stopping ffplay or the ROS node.

USB_PATH="/sys/bus/usb/devices/4-4"

if [ ! -d "$USB_PATH" ]; then
    echo "ERROR: USB device not found at $USB_PATH" >&2
    echo "Check with: ls /sys/bus/usb/devices/ and look for the UGREEN device" >&2
    exit 1
fi

echo "Resetting UGREEN capture card at $USB_PATH..."
echo 0 > "$USB_PATH/authorized"
sleep 0.5
echo 1 > "$USB_PATH/authorized"
sleep 1

if [ -e /dev/video0 ]; then
    echo "OK — /dev/video0 is back"
else
    echo "WARN — /dev/video0 not yet visible, wait a moment and check again"
fi
