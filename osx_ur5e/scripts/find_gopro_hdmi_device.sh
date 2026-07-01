#!/bin/bash
# Print the /dev/video* node for the UGREEN HDMI capture card.
# Usage: find_gopro_hdmi_device.sh [optional_override]
# stdout: device path (for roslaunch <param command="...">)
# stderr: diagnostics and manual instructions on failure

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

_override="${1:-}"

if [[ -n "$_override" ]]; then
    if _has_v4l2_capture_formats "$_override"; then
        printf '%s' "$_override"
        exit 0
    fi
    echo "WARN: override '$_override' is not a v4l2 capture device; auto-detecting..." >&2
fi

USB_PATH=$(find_ugreen_usb_path || true)
if [[ -n "$USB_PATH" ]]; then
    echo "Found UGREEN USB at $USB_PATH ($(cat "$USB_PATH/idVendor"):$(cat "$USB_PATH/idProduct") $(cat "$USB_PATH/product" 2>/dev/null || true))" >&2
else
    echo "WARN: UGREEN USB device not found in sysfs" >&2
fi

DEVICE=$(find_ugreen_capture_device "$USB_PATH" || true)
if [[ -z "$DEVICE" ]]; then
    print_gopro_hdmi_manual_instructions
    exit 1
fi

if ! _has_v4l2_capture_formats "$DEVICE"; then
    echo "WARN: selected $DEVICE without v4l2 format probe (install v4l-utils for verification)" >&2
fi

echo "Using UGREEN capture device: $DEVICE" >&2
# No trailing newline — roslaunch <param command="..."> passes stdout verbatim to usb_cam.
printf '%s' "$DEVICE"
