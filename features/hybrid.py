from features.core import FeatureExtractor
from features.temporal import (
    TemporalGazeFeatureExtractor,
    TemporalPupilFeatureExtractor
)
from features.summary import (
    IPAFeatureGroup,
    LHIPAFeatureGroup
)


class StaticFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(groups=[
            IPAFeatureGroup(),
            LHIPAFeatureGroup()
        ])
