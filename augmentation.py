#AUGMENTATION DE DONNÉES


print("AUGMENTATION DE DONNÉES")

class ArabiziAugmenter:

    def __init__(self):
        # Synonymes Arabizi communs
        self.synonyms = {
            'hmar': ['kelb', '7ayawan', 'behim'],
            'ya3ni': ['يعني', 'y3ni', 'ya3ne'],
            'merde': ['khara', '5ara', 'khra'],
            'wallah': ['wlh', 'walla', 'w allah'],
        }

    def char_swap(self, text, n=2):
        """Échange aléatoire de caractères adjacents"""
        words = text.split()
        for _ in range(n):
            if len(words) > 1:
                idx = np.random.randint(0, len(words))
                word = list(words[idx])
                if len(word) > 2:
                    pos = np.random.randint(0, len(word)-1)
                    word[pos], word[pos+1] = word[pos+1], word[pos]
                    words[idx] = ''.join(word)
        return ' '.join(words)

    def add_typos(self, text):
        # Clavier QWERTY neighbors
        neighbors = {
            'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfcxe',
            'e': 'wsdr', 'f': 'dgcvr', 'g': 'fhbvt', 'h': 'gjbny',
            'i': 'ujko', 'j': 'hkunm', 'k': 'jlmio', 'l': 'kop',
            'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
            'q': 'wa', 'r': 'etdf', 's': 'awedxz', 't': 'ryfg',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
            'y': 'tugh', 'z': 'asx'
        }

        words = text.split()
        if words:
            idx = np.random.randint(0, len(words))
            word = list(words[idx])
            if len(word) > 2:
                pos = np.random.randint(0, len(word))
                if word[pos].lower() in neighbors:
                    word[pos] = np.random.choice(list(neighbors[word[pos].lower()]))
            words[idx] = ''.join(word)
        return ' '.join(words)

    def add_elongation(self, text):
        """Ajoute de l'élongation réaliste"""
        words = text.split()
        if words:
            idx = np.random.randint(0, len(words))
            word = list(words[idx])
            if len(word) > 2:
                pos = np.random.randint(0, len(word))
                word.insert(pos, word[pos] * np.random.randint(2, 4))
            words[idx] = ''.join(word)
        return ' '.join(words)

    def synonym_replacement(self, text):
        """Remplace par des synonymes Arabizi"""
        for word, syns in self.synonyms.items():
            if word in text.lower():
                text = text.replace(word, np.random.choice(syns))
        return text

    def augment(self, text, n_aug=1):
        """Génère n_aug variations du texte"""
        augmented = []
        methods = [self.char_swap, self.add_typos, self.add_elongation, self.synonym_replacement]

        for _ in range(n_aug):
            aug_text = text
            # Appliquer 1-2 méthodes aléatoirement
            for method in np.random.choice(methods, size=np.random.randint(1, 3), replace=False):
                aug_text = method(aug_text)
            augmented.append(aug_text)

        return augmented

# Initialiser l'augmenteur
augmenter = ArabiziAugmenter()

def augment_minority_class(df, target_col='label', minority_class=1,
                           augmentation_factor=2.0):
    """
    Augmente la classe minoritaire
    """
    majority = df[df[target_col] != minority_class]
    minority = df[df[target_col] == minority_class]

    print(f"\nAvant augmentation:")
    print(f"  Classe majoritaire: {len(majority)}")
    print(f"  Classe minoritaire: {len(minority)}")

    # Calculer combien d'exemples générer
    target_size = int(len(majority) / augmentation_factor)
    n_to_generate = max(0, target_size - len(minority))

    print(f"\nObjectif après augmentation: {target_size} exemples")
    print(f"Nombre à générer: {n_to_generate}")

    if n_to_generate == 0:
        return df

    # Générer des exemples augmentés
    augmented_rows = []
    n_per_sample = max(1, n_to_generate // len(minority))

    for idx, row in minority.iterrows():
        aug_texts = augmenter.augment(row['text_clean'], n_aug=n_per_sample)
        for aug_text in aug_texts:
            new_row = row.copy()
            new_row['text_clean'] = aug_text
            new_row['Text'] = aug_text
            augmented_rows.append(new_row)

    # Limiter au nombre voulu
    augmented_rows = augmented_rows[:n_to_generate]
    augmented_df = pd.DataFrame(augmented_rows)

    # Combiner
    df_balanced = pd.concat([df, augmented_df], ignore_index=True)

    print(f"\nAprès augmentation:")
    print(f"  Total: {len(df_balanced)}")
    print(f"  Classe 0: {(df_balanced[target_col] == 0).sum()}")
    print(f"  Classe 1: {(df_balanced[target_col] == 1).sum()}")

    return df_balanced

# Appliquer l'augmentation
df_augmented = augment_minority_class(df, augmentation_factor=2.0)



