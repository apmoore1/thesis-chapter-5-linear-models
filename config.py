from pathlib import Path

PROJECT_DIR = Path(__file__, '..').resolve()
# Datasets
DONG_TRAIN = PROJECT_DIR / 'data' / 'dong' / 'target.train.raw.raw'
DONG_TEST = PROJECT_DIR / 'data' / 'dong' / 'target.test.raw.raw'
ELECTION = PROJECT_DIR / 'data' / 'election'
mitchell_original_train = PROJECT_DIR / 'data' / 'mitchell' / 'en' / '10-fold' / 'train.1'
mitchell_original_test = PROJECT_DIR / 'data' / 'mitchell' / 'en' / '10-fold' / 'test.1'
mitchell_train = PROJECT_DIR / 'data' / 'mitchell_train.xml'
mitchell_test = PROJECT_DIR / 'data' / 'mitchell_test.xml'
youtubean_original = PROJECT_DIR / 'data' / 'samsung_galaxy_s5.xml'
youtubean_train = PROJECT_DIR / 'data' / 'youtubean_train.xml'
youtubean_test = PROJECT_DIR / 'data' / 'youtubean_test.xml'
laptop_train = PROJECT_DIR / 'data' / "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines" / 'Laptop_Train_v2.xml'
laptop_test = PROJECT_DIR / 'data' / 'ABSA_Gold_TestData' / 'Laptops_Test_Gold.xml'
restaurant_train = PROJECT_DIR / 'data' / "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines" / 'Restaurants_Train_v2.xml'
restaurant_test = PROJECT_DIR / 'data' / 'ABSA_Gold_TestData' / 'Restaurants_Test_Gold.xml'

neural_dataset_dir = PROJECT_DIR / 'data' / 'neural'
neural_small_dataset_dir = PROJECT_DIR / 'data' / 'neural_small_training_sets'
small_training_dataset_dir = PROJECT_DIR / 'data' / 'small_training_sets'

# Model Configurations for the Neural Network models
MODEL_CONFIG_DIR = PROJECT_DIR / "model_configs"

# Word Vector Directory
WORD_EMBEDDING_DIR = PROJECT_DIR / 'word embeddings'

# Lexicons
MPQA = PROJECT_DIR / 'data' / 'lexicons' / 'subjectivity_clues_hltemnlp05' / 'subjclueslen1-HLTEMNLP05.tff'
NRC = PROJECT_DIR / 'data' / 'lexicons' / 'NRC-Emotion-Lexicon-v0.92' / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
HL = PROJECT_DIR / 'data' / 'lexicons' / 'opinion-lexicon-English'

# Results and Image directory
RESULTS_DIR = (PROJECT_DIR / 'results').resolve()
IMAGES_DIR = (PROJECT_DIR / 'images').resolve()
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if not IMAGES_DIR.exists():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

