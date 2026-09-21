import os
import tarfile
import shutil
import random
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ARCHIVE_PATH = Path(
    r"C:\Users\kvane\Downloads\256_ObjectCategories.tar"
)

CALTECH_EXTRACT_DIR = (
    PROJECT_DIR / "data" / "caltech256_temp"
)

BIRD_SOURCE_TRAIN = (
    PROJECT_DIR / "data" / "raw" / "train"
)

BIRD_SOURCE_VALID = (
    PROJECT_DIR / "data" / "raw" / "valid"
)

OUTPUT_DIR = (
    PROJECT_DIR / "data" / "binary_v2"
)

OUTPUT_TRAIN_BIRD = (
    OUTPUT_DIR / "train" / "bird"
)

OUTPUT_TRAIN_NONBIRD = (
    OUTPUT_DIR / "train" / "non_bird"
)

OUTPUT_VALID_BIRD = (
    OUTPUT_DIR / "valid" / "bird"
)

OUTPUT_VALID_NONBIRD = (
    OUTPUT_DIR / "valid" / "non_bird"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# We will use a controlled number of images from each
# selected Caltech category.
#
# 60 training + 20 validation per category.
#
# With 30 selected categories this gives approximately:
#
# 30 × 60 = 1,800 non-bird training images
# 30 × 20 =   600 non-bird validation images
#
# This is intentionally smaller than the bird dataset
# for the first V2 experiment.

NONBIRD_TRAIN_PER_CATEGORY = 60
NONBIRD_VALID_PER_CATEGORY = 20


# ============================================================
# SAFE NON-BIRD CATEGORIES
# ============================================================
#
# We deliberately select categories representing:
#
# - vehicles
# - electronics
# - tools
# - household objects
# - food
# - buildings
# - sports equipment
# - instruments
# - miscellaneous objects
#
# We avoid:
#
# - birds
# - animals
# - insects
#
# This reduces ambiguity for the binary classifier.
# ============================================================

SAFE_CATEGORIES = [
    "001.ak47",
    "002.american-flag",
    "003.backpack",
    "004.baseball-bat",
    "005.baseball-glove",
    "006.basketball-hoop",
    "008.bathtub",
    "010.beer-mug",
    "011.billiards",
    "012.binoculars",
    "014.blimp",
    "016.boom-box",
    "017.bowling-ball",
    "018.bowling-pin",
    "019.boxing-glove",
    "020.brain-101",
    "021.breadmaker",
    "023.bulldozer",
    "026.cake",
    "027.calculator",
    "029.cannon",
    "030.canoe",
    "031.car-tire",
    "033.cd",
    "035.cereal-box",
    "036.chandelier-101",
    "037.chess-board",
    "041.coffee-mug",
    "043.coin",
    "045.computer-keyboard",
    "046.computer-monitor",
    "047.computer-mouse",
    "051.cowboy-hat",
    "053.desk-globe",
    "054.diamond-ring",
    "055.dice",
    "058.doorknob",
    "059.drinking-straw",
    "061.dumb-bell",
    "062.eiffel-tower",
    "063.electric-guitar-101",
    "067.eyeglasses",
    "069.fighter-jet",
    "070.fire-extinguisher",
    "071.fire-hydrant",
    "072.fire-truck",
    "074.flashlight",
    "075.floppy-disk",
    "076.football-helmet",
    "077.french-horn",
    "078.fried-egg",
    "079.frisbee",
    "081.frying-pan",
    "082.galaxy",
    "083.gas-pump",
    "088.golf-ball",
    "091.grand-piano-101",
    "092.grapes",
    "094.guitar-pick",
    "095.hamburger",
    "096.hammock",
    "097.harmonica",
    "098.harp",
    "101.head-phones",
    "102.helicopter-101",
    "108.hot-dog",
    "109.hot-tub",
    "110.hourglass",
    "115.ice-cream-cone",
    "117.ipod",
    "120.joy-stick",
    "122.kayak",
    "123.ketch-101",
    "125.knife",
    "126.ladder",
    "127.laptop-101",
    "130.license-plate",
    "131.lightbulb",
    "132.light-house",
    "133.lightning",
    "135.mailbox",
    "136.mandolin",
    "138.mattress",
    "139.megaphone",
    "141.microscope",
    "142.microwave",
    "143.minaret",
    "145.motorbikes-101",
    "146.mountain-bike",
    "148.mussels",
    "149.necktie",
    "153.palm-pilot",
    "155.paperclip",
    "156.paper-shredder",
    "157.pci-card",
    "160.pez-dispenser",
    "161.photocopier",
    "162.picnic-table",
    "163.playing-card",
    "165.pram",
    "167.pyramid",
    "169.radio-telescope",
    "170.rainbow",
    "171.refrigerator",
    "172.revolver-101",
    "173.rifle",
    "174.rotary-phone",
    "175.roulette-wheel",
    "177.saturn",
    "178.school-bus",
    "180.screwdriver",
    "181.segway",
    "182.self-propelled-lawn-mower",
    "183.sextant",
    "184.sheet-music",
    "185.skateboard",
    "187.skyscraper",
    "188.smokestack",
    "191.sneaker",
    "192.snowmobile",
    "193.soccer-ball",
    "194.socks",
    "195.soda-can",
    "196.spaghetti",
    "197.speed-boat",
    "199.spoon",
    "200.stained-glass",
    "202.steering-wheel",
    "203.stirrups",
    "205.superman",
    "206.sushi",
    "208.swiss-army-knife",
    "209.sword",
    "210.syringe",
    "211.tambourine",
    "212.teapot",
    "213.teddy-bear",
    "214.teepee",
    "215.telephone-box",
    "216.tennis-ball",
    "217.tennis-court",
    "218.tennis-racket",
    "219.theodolite",
    "220.toaster",
    "221.tomato",
    "222.tombstone",
    "223.top-hat",
    "224.touring-bike",
    "225.tower-pisa",
    "226.traffic-light",
    "227.treadmill",
    "229.tricycle",
    "230.trilobite-101",
    "231.tripod",
    "232.t-shirt",
    "233.tuning-fork",
    "234.tweezer",
    "235.umbrella-101",
    "237.vcr",
    "238.video-projector",
    "239.washing-machine",
    "240.watch-101",
    "241.waterfall",
    "242.watermelon",
    "243.welding-mask",
    "244.wheelbarrow",
    "245.windmill",
    "246.wine-bottle",
    "247.xylophone",
    "248.yarmulke",
    "249.yo-yo",
    "251.airplanes-101",
    "252.car-side-101",
    "253.faces-easy-101",
    "255.tennis-shoes",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def get_images(folder):
    """Return supported image files from a folder."""

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    if not folder.exists():
        return []

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in extensions
    ]


def create_output_directories():

    directories = [
        OUTPUT_TRAIN_BIRD,
        OUTPUT_TRAIN_NONBIRD,
        OUTPUT_VALID_BIRD,
        OUTPUT_VALID_NONBIRD,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# EXTRACT CALTECH-256
# ============================================================

def extract_caltech():

    print_header("CALTECH-256 EXTRACTION")

    if not ARCHIVE_PATH.exists():

        raise FileNotFoundError(
            f"\nCaltech-256 archive not found:\n"
            f"{ARCHIVE_PATH}"
        )

    extracted_root = (
        CALTECH_EXTRACT_DIR /
        "256_ObjectCategories"
    )

    if extracted_root.exists():

        print(
            "Caltech-256 is already extracted."
        )

        return extracted_root

    CALTECH_EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Archive:\n{ARCHIVE_PATH}"
    )

    print()
    print(
        "Extracting Caltech-256..."
    )

    print(
        "This may take several minutes."
    )

    with tarfile.open(
        ARCHIVE_PATH,
        "r"
    ) as tar:

        tar.extractall(
            CALTECH_EXTRACT_DIR
        )

    if not extracted_root.exists():

        raise RuntimeError(
            "Extraction completed, "
            "but the expected dataset folder "
            "was not found."
        )

    print()
    print(
        "Extraction completed successfully."
    )

    return extracted_root


# ============================================================
# PREPARE BIRD DATA
# ============================================================

def prepare_birds():

    print_header(
        "COPYING EXISTING BIRD DATA"
    )

    if not BIRD_SOURCE_TRAIN.exists():

        raise FileNotFoundError(
            f"Bird training directory not found:\n"
            f"{BIRD_SOURCE_TRAIN}"
        )

    if not BIRD_SOURCE_VALID.exists():

        raise FileNotFoundError(
            f"Bird validation directory not found:\n"
            f"{BIRD_SOURCE_VALID}"
        )

    # Clear only Binary V2 bird folders.
    # Binary V1 is NOT touched.

    if OUTPUT_TRAIN_BIRD.exists():

        shutil.rmtree(
            OUTPUT_TRAIN_BIRD
        )

    if OUTPUT_VALID_BIRD.exists():

        shutil.rmtree(
            OUTPUT_VALID_BIRD
        )

    OUTPUT_TRAIN_BIRD.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_VALID_BIRD.mkdir(
        parents=True,
        exist_ok=True
    )

    train_count = 0
    valid_count = 0

    # --------------------------------------------------------
    # TRAINING BIRDS
    # --------------------------------------------------------

    for class_folder in sorted(
        BIRD_SOURCE_TRAIN.iterdir()
    ):

        if not class_folder.is_dir():
            continue

        images = get_images(
            class_folder
        )

        for image in images:

            destination = (
                OUTPUT_TRAIN_BIRD /
                f"{class_folder.name}__{image.name}"
            )

            shutil.copy2(
                image,
                destination
            )

            train_count += 1

    # --------------------------------------------------------
    # VALIDATION BIRDS
    # --------------------------------------------------------

    for class_folder in sorted(
        BIRD_SOURCE_VALID.iterdir()
    ):

        if not class_folder.is_dir():
            continue

        images = get_images(
            class_folder
        )

        for image in images:

            destination = (
                OUTPUT_VALID_BIRD /
                f"{class_folder.name}__{image.name}"
            )

            shutil.copy2(
                image,
                destination
            )

            valid_count += 1

    print(
        f"Bird training images:   {train_count}"
    )

    print(
        f"Bird validation images: {valid_count}"
    )


# ============================================================
# PREPARE NON-BIRD DATA
# ============================================================

def prepare_non_birds(
    caltech_root
):

    print_header(
        "CREATING REAL-WORLD NON-BIRD DATA"
    )

    random.seed(
        RANDOM_SEED
    )

    available_categories = {}

    for folder in caltech_root.iterdir():

        if folder.is_dir():

            available_categories[
                folder.name
            ] = folder

    print(
        f"Categories in archive: "
        f"{len(available_categories)}"
    )

    selected = [
        category
        for category in SAFE_CATEGORIES
        if category in available_categories
    ]

    print(
        f"Safe categories selected: "
        f"{len(selected)}"
    )

    missing = [
        category
        for category in SAFE_CATEGORIES
        if category not in available_categories
    ]

    if missing:

        print()
        print(
            "WARNING: Some requested "
            "categories were not found:"
        )

        for category in missing:
            print(
                f"  - {category}"
            )

    if len(selected) < 20:

        raise RuntimeError(
            "Too few safe categories were found. "
            "Dataset preparation stopped."
        )

    # Clear previous V2 non-bird data.

    if OUTPUT_TRAIN_NONBIRD.exists():

        shutil.rmtree(
            OUTPUT_TRAIN_NONBIRD
        )

    if OUTPUT_VALID_NONBIRD.exists():

        shutil.rmtree(
            OUTPUT_VALID_NONBIRD
        )

    OUTPUT_TRAIN_NONBIRD.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_VALID_NONBIRD.mkdir(
        parents=True,
        exist_ok=True
    )

    total_train = 0
    total_valid = 0

    # --------------------------------------------------------
    # PROCESS EACH CATEGORY
    # --------------------------------------------------------

    for category in sorted(
        selected
    ):

        category_folder = (
            available_categories[
                category
            ]
        )

        images = get_images(
            category_folder
        )

        random.shuffle(
            images
        )

        required = (
            NONBIRD_TRAIN_PER_CATEGORY
            +
            NONBIRD_VALID_PER_CATEGORY
        )

        if len(images) < required:

            print(
                f"Skipping {category}: "
                f"only {len(images)} images."
            )

            continue

        train_images = images[
            :NONBIRD_TRAIN_PER_CATEGORY
        ]

        valid_images = images[
            NONBIRD_TRAIN_PER_CATEGORY:
            required
        ]

        safe_category_name = (
            category
            .replace(".", "_")
            .replace(" ", "_")
        )

        # ----------------------------------------------------
        # TRAINING IMAGES
        # ----------------------------------------------------

        for index, image in enumerate(
            train_images,
            start=1
        ):

            destination = (
                OUTPUT_TRAIN_NONBIRD /
                f"{safe_category_name}_"
                f"{index:05d}"
                f"{image.suffix.lower()}"
            )

            shutil.copy2(
                image,
                destination
            )

            total_train += 1

        # ----------------------------------------------------
        # VALIDATION IMAGES
        # ----------------------------------------------------

        for index, image in enumerate(
            valid_images,
            start=1
        ):

            destination = (
                OUTPUT_VALID_NONBIRD /
                f"{safe_category_name}_"
                f"{index:05d}"
                f"{image.suffix.lower()}"
            )

            shutil.copy2(
                image,
                destination
            )

            total_valid += 1

    print()
    print(
        f"Non-bird training images:   "
        f"{total_train}"
    )

    print(
        f"Non-bird validation images: "
        f"{total_valid}"
    )

    return total_train, total_valid


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

def show_summary():

    print_header(
        "BINARY V2 DATASET SUMMARY"
    )

    train_bird = len(
        get_images(
            OUTPUT_TRAIN_BIRD
        )
    )

    train_nonbird = len(
        get_images(
            OUTPUT_TRAIN_NONBIRD
        )
    )

    valid_bird = len(
        get_images(
            OUTPUT_VALID_BIRD
        )
    )

    valid_nonbird = len(
        get_images(
            OUTPUT_VALID_NONBIRD
        )
    )

    print(
        f"Train / bird:       {train_bird}"
    )

    print(
        f"Train / non_bird:   {train_nonbird}"
    )

    print(
        f"Valid / bird:       {valid_bird}"
    )

    print(
        f"Valid / non_bird:   {valid_nonbird}"
    )

    print()

    print(
        "Binary V2 location:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "✓ Binary V1 was NOT modified."
    )

    print(
        "✓ binary_test was NOT modified."
    )

    print(
        "✓ The 11 real-world test images "
        "remain completely separate."
    )

    print(
        "✓ No CIFAR-10 images are used "
        "in Binary V2."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_header(
        "BINARY V2 DATASET PREPARATION"
    )

    print(
        f"Project:\n{PROJECT_DIR}"
    )

    print()

    print(
        f"Caltech archive:\n{ARCHIVE_PATH}"
    )

    create_output_directories()

    caltech_root = extract_caltech()

    prepare_birds()

    prepare_non_birds(
        caltech_root
    )

    show_summary()

    print()
    print("=" * 65)
    print(
        "BINARY V2 DATASET PREPARATION COMPLETE"
    )
    print("=" * 65)