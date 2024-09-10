

"""
Eventually, this should be triggered by board comparisons

Functions to add:
- change channel width (narrower/wider)
- change channel depth (shallower/deeper)
-

"""

def change_channel_geometry(channel_geometries, channel_to_change, segments_to_update, change_type="widen"):
    if change_type == "widen":
        key = "b"
        ratio = 1.2
    elif change_type == "narrow":
        key = "b"
        ratio = 0.8
    elif change_type == "deepen":
        key = "H"
        ratio = 1.2
    elif change_type == "undeepen":
        key = "H"
        ratio = 0.8
    else:
        print("Invalid change type given, no changes applied")
        return channel_geometries
    # segments_to_update should always be a list for the below implementation
    for segment in segments_to_update:
        channel_geometries[channel_to_change][key][segment] *= ratio
    return channel_geometries