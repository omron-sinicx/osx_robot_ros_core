#!/bin/bash
# Reset the UGREEN HDMI capture card USB device to recover from UVC URB errors.
# Run this when the capture device stops working after stopping ffplay or the ROS node.

set -euo pipefail

_load_gopro_hdmi_common() {
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$here/gopro_hdmi_common.sh" ]]; then
        # shellcheck source=gopro_hdmi_common.sh
        source "$here/gopro_hdmi_common.sh"
    elif command -v rospack >/dev/null 2>&1; then
        # shellcheck source=gopro_hdmi_common.sh
        source "$(rospack find osx_ur5e)/scripts/gopro_hdmi_common.sh"
    else
        echo "ERROR: gopro_hdmi_common.sh not found" >&2
        exit 1
    fi
}
_load_gopro_hdmi_common

VIDEO_DEVICE_OVERRIDE="${1:-}"

USB_PATH=$(find_ugreen_usb_path || true)
if [[ -z "$USB_PATH" ]]; then
    echo "ERROR: UGREEN HDMI capture USB device not found" >&2
    print_gopro_hdmi_manual_instructions
    exit 1
fi

PRODUCT=$(cat "$USB_PATH/product" 2>/dev/null || echo "UGREEN capture card")
VID=$(cat "$USB_PATH/idVendor")
PID=$(cat "$USB_PATH/idProduct")

echo "Resetting $PRODUCT at $USB_PATH (${VID}:${PID})..."
echo 0 > "$USB_PATH/authorized"
sleep 0.5
echo 1 > "$USB_PATH/authorized"
sleep 1

DEVICE=$(find_ugreen_capture_device "$USB_PATH" || true)
if [[ -z "$DEVICE" ]]; then
    echo "ERROR: UGREEN USB reset succeeded but no /dev/video* node appeared" >&2
    print_gopro_hdmi_manual_instructions
    exit 1
fi

if [[ -n "$VIDEO_DEVICE_OVERRIDE" && "$VIDEO_DEVICE_OVERRIDE" != "$DEVICE" ]]; then
    echo "WARN: launch arg video_device=$VIDEO_DEVICE_OVERRIDE differs from detected $DEVICE" >&2
    echo "      Update gopro_bringup.launch or pass: video_device:=$DEVICE" >&2
fi

if _has_v4l2_capture_formats "$DEVICE"; then
    echo "OK — capture device $DEVICE is ready (Video Capture + formats verified)"
else
    echo "OK — capture device $DEVICE is present (install v4l-utils to verify formats)"
fi

echo "If usb_cam fails to open the device, set in gopro_bringup.launch:" >&2
echo "  <arg name=\"video_device\" default=\"$DEVICE\"/>" >&2
echo "or launch with: roslaunch osx_ur5e gopro_bringup.launch video_device:=$DEVICE" >&2
