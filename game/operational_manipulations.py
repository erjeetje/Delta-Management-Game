

def update_operational_rules(operational_df_old, hexagon_tracker):
    operational_df = operational_df_old.copy()
    operational_df["red_markers"] = operational_df.apply(
        lambda row: hexagon_tracker.loc[row.name, "red_markers"], axis=1)
    operational_df["red_changed"] = operational_df.apply(
        lambda row: hexagon_tracker.loc[row.name, "red_changed"], axis=1)

    def update_operational_buttons(button, red_markers):
        if button == "Qhar":
            if red_markers == 0:
                return 25
            elif red_markers == 1:
                return 50
            elif red_markers == 2:
                return 75
            elif red_markers == 3:
                return 100
        elif button == "Qhar_threshold":
            if red_markers == 0:
                return 900
            elif red_markers == 1:
                return 1100
            elif red_markers == 2:
                return 1300
            elif red_markers == 3:
                return 1500
        elif button == "Qhag":
            if red_markers == 0:
                return 25
            elif red_markers == 1:
                return 35
            elif red_markers == 2:
                return 45
            elif red_markers == 3:
                return 55
        elif button == "Qhij":
            if red_markers == 0:
                return 2
            elif red_markers == 1:
                return 4
            elif red_markers == 2:
                return 6
            elif red_markers == 3:
                return 8
        elif button == "Qhij_threshold":
            if red_markers == 0:
                return 800
            elif red_markers == 1:
                return 1000
            elif red_markers == 2:
                return 1200
            elif red_markers == 3:
                return 1400

    operational_df['Qvalue'] = operational_df.apply(
        lambda row: row['Qvalue'] if not row["red_changed"] else update_operational_buttons(
            row["Qtype"], row["red_markers"]), axis=1)
    return operational_df