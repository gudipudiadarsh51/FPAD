from afqa_toolbox.features import FeatRPS                             # type: ignore # Radial Power Spectrum


def compute_rps(foreground_cropped):
    rps_feat = FeatRPS()
    rps_value = rps_feat.rps(foreground_cropped)

    return float(rps_value)
