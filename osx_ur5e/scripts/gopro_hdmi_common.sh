#!/bin/bash
# Shared helpers for UGREEN GoPro HDMI capture card discovery and reset.

# Known UGREEN capture cards (MACROSILICON / ITE chipsets).
readonly _UGREEN_KNOWN_IDS=(
    "345f:2131"   # UGREEN 25773
    "3188:1000"   # UGREEN 25173
    "2b89:5389"   # UGREEN 15389
)

print_gopro_hdmi_manual_instructions() {
    cat >&2 <<'EOF'
Could not find the UGREEN HDMI capture device automatically.

Manual check:
  1. lsusb | grep -i ugreen
  2. v4l2-ctl --list-devices
     Look for the "UGREEN ..." block and note its /dev/video* nodes.

  3. For each candidate node, find the one that actually captures video:
       v4l2-ctl -d /dev/videoX --all | grep -E 'Device Caps|Video Capture'
       v4l2-ctl -d /dev/videoX --list-formats-ext

     The capture node shows "Video Capture" in Device Caps and formats like YUYV/MJPG.
     Metadata-only nodes (often the higher /dev/video number) have no pixel formats.

  4. Pass the working device to the launch file:
       roslaunch osx_ur5e gopro_bringup.launch video_device:=/dev/videoX

  Or run the helper directly:
       rosrun osx_ur5e find_gopro_hdmi_device.sh
EOF
}

_is_known_ugreen_id() {
    local vid="$1" pid="$2" known
    for known in "${_UGREEN_KNOWN_IDS[@]}"; do
        [[ "${vid}:${pid}" == "$known" ]] && return 0
    done
    return 1
}

find_ugreen_usb_path() {
    local dev product vid pid
    for dev in /sys/bus/usb/devices/*; do
        [[ -f "$dev/idVendor" && -f "$dev/authorized" ]] || continue
        product=$(cat "$dev/product" 2>/dev/null || true)
        vid=$(cat "$dev/idVendor" 2>/dev/null || true)
        pid=$(cat "$dev/idProduct" 2>/dev/null || true)
        if [[ "$product" == *"UGREEN"* ]] || _is_known_ugreen_id "$vid" "$pid"; then
            echo "$dev"
            return 0
        fi
    done
    return 1
}

_video_belongs_to_usb() {
    local dev="$1" usb_path="$2"
    local video_sysfs="/sys/class/video4linux/$(basename "$dev")"
    [[ -d "$video_sysfs" ]] || return 1
    local video_usb
    video_usb=$(readlink -f "$video_sysfs/device/../../" 2>/dev/null || true)
    [[ -n "$video_usb" && "$video_usb" == "$(readlink -f "$usb_path")" ]]
}

_has_v4l2_capture_formats() {
    local dev="$1" formats
    command -v v4l2-ctl >/dev/null 2>&1 || return 1
    formats=$(v4l2-ctl -d "$dev" --list-formats 2>/dev/null) || return 1
    echo "$formats" | grep -q "Type: Video Capture" || return 1
    echo "$formats" | grep -qE 'YUYV|MJPG|NV12|BGR3' || return 1
    return 0
}

find_ugreen_capture_device() {
    local usb_path="$1" dev video_sysfs index best_dev="" best_index=999
    local -a candidates=()

    if [[ -n "$usb_path" ]]; then
        for video_sysfs in /sys/class/video4linux/video*; do
            dev="/dev/$(basename "$video_sysfs")"
            _video_belongs_to_usb "$dev" "$usb_path" || continue
            candidates+=("$dev")
        done
    fi

    if [[ ${#candidates[@]} -eq 0 ]]; then
        local vid pid product
        for video_sysfs in /sys/class/video4linux/video*; do
            dev="/dev/$(basename "$video_sysfs")"
            vid=$(cat "$video_sysfs/device/../idVendor" 2>/dev/null || true)
            pid=$(cat "$video_sysfs/device/../idProduct" 2>/dev/null || true)
            product=$(cat "$video_sysfs/name" 2>/dev/null || true)
            if [[ "$product" == *"UGREEN"* ]] || _is_known_ugreen_id "$vid" "$pid"; then
                candidates+=("$dev")
            fi
        done
    fi

    for dev in "${candidates[@]}"; do
        _has_v4l2_capture_formats "$dev" || continue
        index=$(cat "/sys/class/video4linux/$(basename "$dev")/index" 2>/dev/null || echo 999)
        if (( index < best_index )); then
            best_index=$index
            best_dev=$dev
        fi
    done

    if [[ -n "$best_dev" ]]; then
        echo "$best_dev"
        return 0
    fi

    # Fall back to lowest-index UGREEN node when v4l2-ctl is unavailable.
    best_dev=""
    best_index=999
    for dev in "${candidates[@]}"; do
        index=$(cat "/sys/class/video4linux/$(basename "$dev")/index" 2>/dev/null || echo 999)
        if (( index < best_index )); then
            best_index=$index
            best_dev=$dev
        fi
    done
    [[ -n "$best_dev" ]] && echo "$best_dev"
}
