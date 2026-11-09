"""배경음 카테고리. 코드는 DB(background_sounds.category)와 web 표시 이름의 기준이다.

labels는 FSD50K 정답 라벨(AudioSet 이름) 또는 FMA 장르 이름이다. 먼저 맞는 카테고리 하나에 넣는다(순서가 우선순위).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    code: str
    kind: str  # AMBIENCE(장소·환경음), EFFECT(효과음), MUSIC(배경음악)
    label: str
    labels: frozenset[str]
    keywords: tuple[str, ...] = ()
    keyword_only: bool = False
    all_labels: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    min_seconds: float = 0.0
    max_seconds: float = 0.0
    prompt: str = ""  # CLAP 확인용 영어 문장


def _c(code: str, kind: str, label: str, labels: str, prompt: str, **kw: object) -> Category:
    return Category(
        code, kind, label, frozenset(x for x in labels.split() if x), prompt=prompt, **kw
    )  # type: ignore[arg-type]


SPEECH = frozenset({"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking"})
MUSICAL = frozenset({"Music", "Musical_instrument"})
A, E, M = "AMBIENCE", "EFFECT", "MUSIC"
AMB = {"min_seconds": 8.0}

AMBIENCE = [
    _c(
        "SCHOOL_KIDS",
        A,
        "학교에서 아이들 떠드는 소리",
        "Child_speech_and_kid_speaking",
        "children talking and playing at a school playground",
        keywords=("school", "playground", "children", "kids", "schoolyard", "recess"),
        **AMB,
    ),
    _c(
        "OCEAN_WAVES",
        A,
        "바닷가 파도",
        "Waves_and_surf Ocean",
        "ocean waves crashing on a beach",
        keywords=("beach", "sea", "waves", "surf", "ocean"),
        **AMB,
    ),
    _c(
        "THUNDERSTORM",
        A,
        "천둥 번개",
        "Thunder Thunderstorm",
        "thunderstorm with rolling thunder",
        **AMB,
    ),
    _c(
        "RAIN",
        A,
        "빗소리",
        "Rain Raindrop",
        "steady rain falling",
        exclude=frozenset({"Thunder"}),
        **AMB,
    ),
    _c(
        "NIGHT_INSECTS",
        A,
        "밤 귀뚜라미·풀벌레",
        "Cricket Insect",
        "crickets chirping on a summer night",
        keywords=("night", "crickets", "cicada", "summer"),
        **AMB,
    ),
    _c("FROGS", A, "개구리 우는 논", "Frog", "frogs croaking by a pond", **AMB),
    _c(
        "STREAM",
        A,
        "시냇물·계곡",
        "Stream Trickle_and_dribble Gurgling",
        "a babbling brook in a valley",
        **AMB,
    ),
    _c(
        "FOREST_BIRDS",
        A,
        "숲속 새소리",
        "Bird_vocalization_and_bird_call_and_bird_song Chirp_and_tweet Bird",
        "birds singing in a forest in the morning",
        exclude=frozenset({"Gull_and_seagull"}),
        **AMB,
    ),
    _c(
        "HARBOR",
        A,
        "항구·갈매기",
        "Gull_and_seagull Boat_and_Water_vehicle",
        "seagulls and boats at a harbor",
        **AMB,
    ),
    _c("WIND", A, "바람 소리", "Wind", "strong wind blowing", **AMB),
    _c(
        "CAMPFIRE",
        A,
        "모닥불",
        "Fire Crackle",
        "a crackling campfire",
        exclude=frozenset({"Fireworks"}),
        **AMB,
    ),
    _c(
        "CAFE",
        A,
        "카페·식당",
        "Cutlery_and_silverware Dishes_and_pots_and_pans Chink_and_clink",
        "a busy cafe with people chatting and dishes clinking",
        all_labels=frozenset(),
        keywords=("cafe", "restaurant", "coffee", "bar", "pub"),
        min_seconds=8.0,
    ),
    _c(
        "CHEERING",
        A,
        "환호·박수",
        "Cheering Applause Clapping",
        "a crowd cheering and applauding",
        min_seconds=5.0,
    ),
    _c(
        "PARTY_CROWD",
        A,
        "사람들 웅성거림",
        "Chatter Crowd Conversation",
        "a crowd of people talking",
        **AMB,
    ),
    _c(
        "CITY_TRAFFIC",
        A,
        "도시 거리·자동차",
        "Traffic_noise_and_roadway_noise Car_passing_by Motor_vehicle_(road)",
        "city street traffic with cars passing",
        **AMB,
    ),
    _c(
        "SUBWAY_TRAIN",
        A,
        "지하철·기차",
        "Subway_and_metro_and_underground Train Rail_transport",
        "a train passing on railway tracks",
        **AMB,
    ),
    _c(
        "AIRPLANE",
        A,
        "공항·비행기",
        "Fixed-wing_aircraft_and_airplane Aircraft",
        "an airplane flying overhead",
        **AMB,
    ),
    _c(
        "KITCHEN",
        A,
        "부엌 요리",
        "Frying_(food) Boiling",
        "food sizzling in a frying pan in a kitchen",
        min_seconds=6.0,
    ),
    _c(
        "OFFICE",
        A,
        "사무실 타자 소리",
        "Typing Computer_keyboard Typewriter Printer",
        "typing on a keyboard in an office",
        min_seconds=6.0,
    ),
    _c(
        "CLOCK",
        A,
        "째깍째깍 시계",
        "Tick-tock Clock Tick",
        "a clock ticking in a quiet room",
        min_seconds=6.0,
    ),
    _c(
        "FARM",
        A,
        "시골 농장",
        "Chicken_and_rooster Livestock_and_farm_animals_and_working_animals Fowl",
        "farm animals in the countryside",
        min_seconds=5.0,
    ),
    _c(
        "CHURCH_BELLS",
        A,
        "종소리",
        "Church_bell Bell Chime",
        "church bells ringing",
        min_seconds=5.0,
    ),
    _c(
        "WIND_CHIME",
        A,
        "풍경 소리",
        "Wind_chime",
        "wind chimes in a gentle breeze",
        min_seconds=4.0,
    ),
    _c(
        "FIREWORKS",
        A,
        "불꽃놀이",
        "Fireworks",
        "fireworks exploding at a festival",
        min_seconds=4.0,
    ),
    _c(
        "FOOTSTEPS",
        A,
        "걷는 발소리",
        "Walk_and_footsteps Run",
        "footsteps walking",
        min_seconds=5.0,
    ),
    _c(
        "WATER_HOME",
        A,
        "물 흐르는 소리",
        "Water_tap_and_faucet Sink_(filling_or_washing) "
        "Bathtub_(filling_or_washing) Pour Fill_(with_liquid)",
        "water running from a tap",
        min_seconds=5.0,
    ),
    _c(
        "PETS",
        A,
        "강아지·고양이",
        "Dog Bark Cat Meow Purr",
        "a dog barking and a cat meowing",
        min_seconds=4.0,
    ),
]

EFFECT = [
    _c(
        "DOOR_KNOCK",
        E,
        "노크·문 소리",
        "Knock Door Slam Sliding_door Cupboard_open_or_close",
        "knocking on a door",
    ),
    _c("DOORBELL", E, "초인종", "Doorbell", "a doorbell ringing"),
    _c("PHONE", E, "전화벨", "Telephone Ringtone", "a telephone ringing"),
    _c("GLASS_BREAK", E, "깨지는 소리", "Shatter Glass Crash_cymbal", "glass shattering"),
    _c("EXPLOSION", E, "폭발", "Explosion Boom Gunshot_and_gunfire", "a loud explosion"),
    _c("WHOOSH", E, "휙 지나가는 소리", "Whoosh_and_swoosh_and_swish", "a fast whoosh"),
    _c("LAUGHTER", E, "웃음소리", "Laughter Giggle Chuckle_and_chortle", "people laughing"),
    _c("SCREAM", E, "비명·고함", "Screaming Yell Shout", "a person screaming"),
    _c("CRYING", E, "우는 소리", "Crying_and_sobbing", "a person crying and sobbing"),
    _c("GASP_SIGH", E, "한숨·헉", "Gasp Sigh Breathing", "a person gasping and sighing"),
    _c("CAMERA", E, "카메라 셔터", "Camera", "a camera shutter click"),
    _c("SIREN", E, "사이렌", "Siren", "an ambulance siren"),
    _c("CAR_HORN", E, "자동차 경적", "Vehicle_horn_and_car_horn_and_honking", "a car horn honking"),
    _c(
        "FUNNY",
        E,
        "우스꽝스러운 소리",
        "Burping_and_eructation Fart Squeak Boing",
        "a funny cartoon sound",
    ),
]

FMA_GENRES = [
    _c(
        "MUSIC_AMBIENT",
        M,
        "잔잔한 앰비언트",
        "Ambient Ambient_Electronic New_Age Drone",
        "calm ambient background music",
    ),
    _c(
        "MUSIC_SOUNDTRACK",
        M,
        "영화 같은 음악",
        "Soundtrack Cinematic",
        "cinematic soundtrack music",
    ),
    _c("MUSIC_PIANO", M, "피아노", "Piano", "solo piano music"),
    _c(
        "MUSIC_CLASSICAL",
        M,
        "클래식",
        "Classical Chamber_Music Symphony Opera Baroque Romantic Choral_Music",
        "classical orchestral music",
    ),
    _c(
        "MUSIC_JAZZ",
        M,
        "재즈",
        "Jazz Jazz:_Vocal Free-Jazz Big_Band/Swing Bebop",
        "jazz music with saxophone",
    ),
    _c(
        "MUSIC_CHIPTUNE",
        M,
        "게임 음악(8비트)",
        "Chiptune Chip_Music",
        "8-bit chiptune video game music",
    ),
    _c(
        "MUSIC_ELECTRONIC",
        M,
        "일렉트로닉",
        "Electronic Techno House Dance Dubstep Electro-Punk "
        "Glitch Downtempo IDM Trip-Hop Breakcore_-_Hard Drum_&_Bass",
        "electronic dance music",
    ),
    _c("MUSIC_HIPHOP", M, "힙합 비트", "Hip-Hop Hip-Hop_Beats Rap", "hip hop beat"),
    _c(
        "MUSIC_ROCK",
        M,
        "록",
        "Rock Punk Metal Indie-Rock Post-Rock Garage Psych-Rock Loud-Rock",
        "rock music with electric guitar",
    ),
    _c("MUSIC_POP", M, "팝", "Pop Synth_Pop Experimental_Pop Power-Pop", "upbeat pop music"),
    _c(
        "MUSIC_FOLK",
        M,
        "포크·어쿠스틱",
        "Folk Free-Folk Singer-Songwriter Psych-Folk British_Folk",
        "acoustic folk guitar music",
    ),
    _c(
        "MUSIC_WORLD",
        M,
        "세계 음악",
        "International Latin_America African Asia-Far_East Middle_East "
        "Brazilian Celtic Balkan Indian Reggae_-_Dub",
        "traditional world music",
    ),
    _c(
        "MUSIC_RETRO",
        M,
        "옛날 음악",
        "Blues Old-Time_/_Historic Country Soul-RnB Easy_Listening Lounge Swing",
        "old fashioned retro music",
    ),
    _c("MUSIC_INSTRUMENTAL", M, "연주곡", "Instrumental", "instrumental music"),
    _c(
        "MUSIC_EXPERIMENTAL",
        M,
        "실험적인 음악",
        "Experimental Noise Sound_Collage Avant-Garde Musique_Concrete",
        "experimental abstract sound art",
    ),
]

ALL = AMBIENCE + EFFECT + FMA_GENRES

# 분위기 태그 (DB background_sounds.moods). CLAP으로 소리에 맞는지 재고, 설명을 쓸 때 다시 고른다
MOODS: dict[str, tuple[str, str]] = {
    "PEACEFUL": ("평화로운", "peaceful and calm"),
    "JOYFUL": ("즐거운", "happy and cheerful"),
    "EXCITING": ("신나는", "exciting and energetic"),
    "PARTY": ("파티", "festive party atmosphere"),
    "TENSE": ("긴장되는", "tense and suspenseful"),
    "SCARY": ("무서운", "scary and creepy"),
    "SAD": ("슬픈", "sad and melancholic"),
    "SOLEMN": ("엄숙한", "solemn and serious"),
    "MYSTERIOUS": ("신비로운", "mysterious and dreamy"),
    "EPIC": ("웅장한", "epic and grand"),
    "ROMANTIC": ("로맨틱한", "romantic and warm"),
    "COMIC": ("코믹한", "funny and playful"),
    "LONELY": ("쓸쓸한", "lonely and empty"),
    "NOSTALGIC": ("그리운", "nostalgic and sentimental"),
}
