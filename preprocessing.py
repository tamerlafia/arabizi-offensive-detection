#PRÉTRAITEMENT

print("PARTIE 1: PRÉTRAITEMENT AVANCÉ AVEC NORMALISATION ARABIZI")

class ArabiziNormalizer:


    def __init__(self):
        # Mapping des numéros Arabizi vers lettres arabes
        self.number_mapping = {
            '2': 'a',   # همزة
            '3': 'aa',  # ع
            '5': 'kh',  # خ
            '6': 't',   # ط
            '7': 'h',   # ح
            '8': 'gh',  # غ
            '9': 'q',   # ق
        }

        # Variations communes Arabizi
        self.arabizi_variations = {
            'kh': ['5', 'kh'],
            'gh': ['8', 'gh'],
            'sh': ['ch', 'sh'],
            'aa': ['3', 'aa', 'a3'],
        }

        # Mots offensifs communs
        self.offensive_patterns = [
            r'\bhmar\b', r'\bkelb\b', r'\bkalb\b', r'\bwati\b',
            r'\bhayawan\b', r'\bmanyak\b', r'\bkhara\b'
        ]

    def remove_elongation(self, text):
        """réduit l'élongation: looool -> lol"""
        # Remplace 3+ caractères répétés par 2 occurrences
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def map_numbers_to_letters(self, text):
        for num, letter in self.number_mapping.items():
            text = text.replace(num, letter)
        return text

    def normalize_arabizi(self, text):
        # Minuscules
        text = text.lower()

        # Suppression élongation
        text = self.remove_elongation(text)

        # Mapping numéros -> lettres
        text = self.map_numbers_to_letters(text)



    def detect_offensive_intensity(self, text):
        score = 0.0
        for pattern in self.offensive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sc0ore += .2

        # Vérifier les majuscules excessives (CRIS = agressivité)
        if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.5:
            score += 0.1

        # Ponctuation excessive (!!!, ???)
        if len(re.findall(r'[!?]{2,}', text)) > 0:
            score += 0.1

        return min(score, 1.0)

normalizer = ArabiziNormalizer()

def advanced_preprocess(text, keep_emojis=True, normalize_arabizi=True):

    original_text = text

    # Conversion minuscule
    text = text.lower()

    # Suppression URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)

    # Suppression mentions
    text = re.sub(r'@\w+', '[USER]', text)

    # Gestion emojis
    emoji_list = []
    if keep_emojis:
        for char in text:
            if char in emoji.EMOJI_DATA:
                emoji_list.append(char)
        text = emoji.demojize(text, language='en')
    else:
        text = emoji.replace_emoji(text, '')

    # Normalisation Arabizi
    if normalize_arabizi:
        text = normalizer.normalize_arabizi(text)

    # Suppression ponctuation excessive
    text = re.sub(r'([!?.]){2,}', r'\1', text)

    # Suppression ponctuation non pertinente
    text = re.sub(r'[^\w\s\':_!?]', ' ', text)

    # Espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text, emoji_list

def detect_language_mix(text):
    """Détecte le mélange de langues"""
    try:
        langs = detect_langs(text)
        return [(lang.lang, lang.prob) for lang in langs]
    except:
        return [('unknown', 1.0)]

def extract_features(text, original_text):
    """Extraction de features linguistiques avancées"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'uppercase_ratio': sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1),
        'exclamation_count': original_text.count('!'),
        'question_count': original_text.count('?'),
        'emoji_count': len([c for c in original_text if c in emoji.EMOJI_DATA]),
        'mention_count': len(re.findall(r'@\w+', original_text)),
        'url_count': len(re.findall(r'http\S+', original_text)),
        'number_count': sum(c.isdigit() for c in original_text),
        'offensive_intensity': normalizer.detect_offensive_intensity(text),
    }

    # Détection de langues
    lang_mix = detect_language_mix(original_text)
    features['primary_language'] = lang_mix[0][0] if lang_mix else 'unknown'
    features['language_diversity'] = len(lang_mix)

    return features

# Charger les données
df=pd.read_csv("Arabizi-Off_Lang_Dataset.csv")

print("\nprétraitement")

# Appliquer prétraitement
df['text_clean'] = df['Text'].apply(lambda x: advanced_preprocess(x, normalize_arabizi=True)[0])
df['emojis'] = df['Text'].apply(lambda x: advanced_preprocess(x, normalize_arabizi=True)[1])

# Extraire features
feature_dicts = df.apply(lambda row: extract_features(row['text_clean'], row['Text']), axis=1)
feature_df = pd.DataFrame(list(feature_dicts))
df = pd.concat([df, feature_df], axis=1)

# Exemples de normalisation
print("\nExemples de normalisation Arabizi:")
for i in range(min(5, len(df))):
    print(f"\n{i+1}. Original: {df['Text'].iloc[i]}")
    print(f"   Normalisé: {df['text_clean'].iloc[i]}")
    print(f"   Features: Length={df['length'].iloc[i]}, Offensive={df['offensive_intensity'].iloc[i]:.2f}")

# Encoder labels
df['label'] = df['Generic Class'].map({'non-offensive': 0, 'offensive': 1})

# Statistiques
print("\nStatistiques")
print(f"Distribution des classes:")
print(df['label'].value_counts())
print(f"\nProportion: {df['label'].value_counts(normalize=True)}")


