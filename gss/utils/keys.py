"""
The keys file is part of the new database concept 2017.

These are database related keys. Use them in your database JSON.

Please avoid abbreviations and discuss first, before adding new keys.
"""
DATASETS = "datasets"
DATASET_NAME = "dataset"
EXAMPLES = "examples"
META = "meta"
ALIAS = "alias"

# Information per example
AUDIO_PATH = "audio_path"
AUDIO_LENGTH = "audio_length"
AUDIO_DATA = "audio_data"
NUM_SAMPLES = "num_samples"
NUM_FRAMES = "num_frames"
NUM_SPEAKERS = "num_speakers"
EXAMPLE_ID = "example_id"  # Replaces mixture ID for multi-speaker scenario.
SPEAKER_ID = "speaker_id"
GENDER = "gender"
START = "start"  # startingtime for speaker_id
END = "end"  # endingtime for speaker_id

# Segmentation refers to a list of tuples
# [(<label 1>, <start sample>, <end sample>), ...]
SEGMENTATION = "segmentation"
# Transcription refers to a list of labels [<label 1>, <label 2>, ...] providing
# the labels that appear in an example in a certain order
TRANSCRIPTION = "transcription"
KALDI_TRANSCRIPTION = "kaldi_transcription"
# Tags refers to a list of labels [<label 1>, <label 2>, ...] providing
# the labels that appear in an example in a any order
TAGS = "tags"
# Transcription refers to a list of states [<s1>, <s2>, ...] providing
# the state alignment sequence usually inferred by Kaldi
ALIGNMENT = "alignment"
NUM_ALIGNMENT_FRAMES = "num_alignment_frames"

# Signals
OBSERVATION = "observation"
LAPEL = "lapel"
HEADSET = "headset"

SPEECH_SOURCE = "speech_source"
SPEECH_IMAGE = "speech_image"
NOISE_IMAGE = "noise_image"
LOG_WEIGHTS = "log_weights"  # Stupid name. Should be e.g. ratio_decibel.

# SPEECH_REVERBARATON
SPEECH_REVERBERATION_DIRECT = "speech_reverberation_direct"
SPEECH_REVERBERATION_TAIL = "speech_reverberation_tail"

SOUND_DECAY_TIME = "sound_decay_time"  # Also referred to as T60 time.
RIR = "rir"
RIR_DIRECT = "rir_direct"
RIR_TAIL = "rir_tail"

# Dimension prefixes for i.e. observation signal.
ARRAY = "a"
SPEAKER = "s"
CHANNEL = "c"

# other sub-keys
MALE = "male"
FEMALE = "female"
PHONES = "phones"
WORDS = "words"
EVENTS = "events"
SAMPLE_RATE = "sample_rate"

# temporary keys, need to be discussed
LABELS = "labels"
MAPPINGS = "mappings"

# Scenario related keys
LOCATION = "location"
ROOM_DIMENSIONS = "room_dimensions"
SOURCE_POSITION = "source_position"
SENSOR_POSITION = "sensor_position"
