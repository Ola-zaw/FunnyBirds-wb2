import json
import random
import math
import os

# =====================================================================
# BAZA WIEDZY (KNOWLEDGE BASE)
# =====================================================================
KNOWLEDGE_BASE = {
    'beak_model': ['beak01.glb', 'beak02.glb', 'beak03.glb', 'beak04.glb'], 
    'beak_color': ['yellow', 'black', 'red', 'green'], 
    'eye_model':  ['eye01.glb', 'eye02.glb', 'eye03.glb'],
    'foot_model': ['foot01.glb', 'foot02.glb', 'foot03.glb', 'foot04.glb'],
    'tail_model': ['tail01.glb', 'tail02.glb', 'tail03.glb', 'tail04.glb', 'tail05.glb', 'tail06.glb', 'tail07.glb', 'tail08.glb', 'tail09.glb'],
    'tail_color': ['red', 'green', 'blue'],
    'wing_model': ['wing01.glb', 'wing02.glb', 'wing03.glb', 'wing04.glb', 'wing05.glb', 'wing06.glb'],
    'wing_color': ['red', 'green', 'blue', 'yellow']
}

# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================
def get_frozen_defaults(frozen_features_config):
    """Zwraca domyślne (zamrożone) wartości dla cech, które nie są modyfikowane."""
    frozen = {}
    for feature, index in frozen_features_config.items():
        if index == -1:
            frozen[feature] = KNOWLEDGE_BASE[feature][0]
        else:
            frozen[feature] = KNOWLEDGE_BASE[feature][index]
    return frozen

def generate_camera_and_bg(sample):
    """Dodaje tło i parametry kamery do próbki (analogicznie do starego framework.py)."""
    sample['camera_distance'] = random.randint(200, 400)
    sample['camera_pitch'] = random.uniform(0, 2*math.pi)
    sample['camera_roll'] = random.uniform(0, 2*math.pi)
    sample['light_distance'] = 300
    sample['light_pitch'] = random.uniform(0, 2*math.pi)
    sample['light_roll'] = random.uniform(0, 2*math.pi)

    nr_bg_parts = random.randint(10, 30)
    sample['bg_objects'] = "".join([str(random.randint(0, 4)) + "," for _ in range(nr_bg_parts)])
    sample['bg_radius'] = "".join([str(random.randint(100, 200)) + "," for _ in range(nr_bg_parts)])
    sample['bg_pitch'] = "".join([str(random.uniform(0, 2*math.pi)) + "," for _ in range(nr_bg_parts)])
    sample['bg_roll'] = "".join([str(random.uniform(0, 2*math.pi)) + "," for _ in range(nr_bg_parts)])
    sample['bg_scale_x'] = "".join([str(random.randint(5, 20)) + "," for _ in range(nr_bg_parts)])
    sample['bg_scale_y'] = "".join([str(random.randint(5, 20)) + "," for _ in range(nr_bg_parts)])
    sample['bg_scale_z'] = "".join([str(random.randint(5, 20)) + "," for _ in range(nr_bg_parts)])
    sample['bg_rot_x'] = "".join([str(random.uniform(0, 2*math.pi)) + "," for _ in range(nr_bg_parts)])
    sample['bg_rot_y'] = "".join([str(random.uniform(0, 2*math.pi)) + "," for _ in range(nr_bg_parts)])
    sample['bg_rot_z'] = "".join([str(random.uniform(0, 2*math.pi)) + "," for _ in range(nr_bg_parts)])
    sample['bg_color'] = "".join([random.choice(['red', 'green', 'blue', 'yellow']) + "," for _ in range(nr_bg_parts)])
    return sample

def save_dataset(dataset, root_path, mode="train"):
    """Zapisuje wygenerowany dataset do JSON w odpowiednim folderze."""
    target_dir = os.path.join(root_path, "FunnyBirds")
    os.makedirs(target_dir, exist_ok=True) 
    
    file_name = f"dataset_{mode}.json"
    full_path = os.path.join(target_dir, file_name)

    with open(full_path, "w") as outfile:
        json.dump(dataset, outfile)
    print(f"Zapisano {len(dataset)} ptaków bezpośrednio do folderu: {full_path}")

# =====================================================================
# METODA 1: GENERACJA PROBABILISTYCZNA
# =====================================================================
def generate_probabilistic_dataset(prob_config, frozen_config, num_classes, samples_per_class, root_path, mode="train"):
    """
    Generuje ptaki na podstawie zadanego prawdopodobieństwa występowania wariantów dla każdej klasy.
    """
    dataset = []
    frozen_features = get_frozen_defaults(frozen_config)

    for class_id in range(num_classes):
        class_rules = prob_config.get(class_id, {})
        
        for _ in range(samples_per_class):
            sample = {'class_idx': class_id}
            
            # Wypełnij zamrożonymi cechami
            for key, val in frozen_features.items():
                sample[key] = val
                
            # Wypełnij cechami probabilistycznymi zdefiniowanymi dla tej klasy
            for feature, rule in class_rules.items():
                options = rule['options']
                probs = rule['probs']
                
                # Losowanie z wagami (prawdopodobieństwami)
                chosen_idx = random.choices(options, weights=probs, k=1)[0]
                sample[feature] = KNOWLEDGE_BASE[feature][chosen_idx]

            # Wylosuj pozostale (niezamrozone) cechy
            for feature in KNOWLEDGE_BASE.keys():
                if feature not in sample:
                    sample[feature] = random.choice(KNOWLEDGE_BASE[feature])

            sample = generate_camera_and_bg(sample)
            dataset.append(sample)
            
    save_dataset(dataset, root_path, mode)

# =====================================================================
# METODA 2: GENERACJA HIERARCHICZNA (ZALEŻNOŚCI)
# =====================================================================
def generate_hierarchical_dataset(hierarchy_rules, frozen_config, num_classes, samples_per_class, root_path, mode="train"):
    """
    Generuje ptaki z zadaną mocą hierarchiczności (zależności) między cechami.
    """
    dataset = []
    frozen_features = get_frozen_defaults(frozen_config)

    for class_id in range(num_classes):
        class_rules = hierarchy_rules.get(class_id, {})
        
        # Jeśli klasa ma zdefiniowane reguły hierarchiczne
        strength = class_rules.get('strength', 0.0) # Szansa (0.0 do 1.0) na aktywację "sztywnej" zależności
        primary_feat = class_rules.get('primary_feature')
        dependent_feat = class_rules.get('dependent_feature')
        forced_combo = class_rules.get('forced_combo') # np. {0: 1} oznacza: jeśli primary_feat wylosuje index 0, to dependent_feat MUSI mieć index 1
        
        for _ in range(samples_per_class):
            sample = {'class_idx': class_id}
            
            for key, val in frozen_features.items():
                sample[key] = val
                
            if class_rules:
                # Losujemy główną cechę całkowicie losowo z dostępnych wariantów (lub można dostosować)
                num_primary_options = len(KNOWLEDGE_BASE[primary_feat])
                primary_idx = random.randint(0, num_primary_options - 1)
                sample[primary_feat] = KNOWLEDGE_BASE[primary_feat][primary_idx]
                
                # Sprawdzamy, czy wpadamy w pułapkę zależności
                apply_dependency = random.random() < strength
                
                if apply_dependency and primary_idx in forced_combo:
                    # Aktywuje się sztywna zależność - wymuszamy przypisanie podrzędnej cechy
                    dependent_idx = forced_combo[primary_idx]
                    sample[dependent_feat] = KNOWLEDGE_BASE[dependent_feat][dependent_idx]
                else:
                    # Losujemy podrzędną cechę niezależnie
                    num_dependent_options = len(KNOWLEDGE_BASE[dependent_feat])
                    dependent_idx = random.randint(0, num_dependent_options - 1)
                    sample[dependent_feat] = KNOWLEDGE_BASE[dependent_feat][dependent_idx]
            
            # Wylosuj pozostale (niezamrozone) cechy
            for feature in KNOWLEDGE_BASE.keys():
                if feature not in sample:
                    sample[feature] = random.choice(KNOWLEDGE_BASE[feature])
            
            sample = generate_camera_and_bg(sample)
            dataset.append(sample)
            
    save_dataset(dataset, root_path, mode)

# =====================================================================
# URUCHOMIENIE I SCENARIUSZE TESTOWE
# =====================================================================
if __name__ == "__main__":
    
    # Wspólne parametry
    NUM_CLASSES = 2
    SAMPLES_PER_CLASS = 1000
    
    # Cechy, które pozostają niezmienne w obu scenariuszach
    frozen_config = {
        'beak_color': -1,
        'foot_model': -1,
        'eye_model':  -1, 
        'tail_model': -1, 
        'wing_model': -1, 
        'wing_color': -1
    }

    # # =================================================================
    # # POPRAWIONY KONFIG DLA HIERARCHII
    # # Mrozimy wszystko OPRÓCZ koloru skrzydeł i koloru dzioba
    # # =================================================================
    # frozen_config_hier = {
    #     'beak_model': -1,
    #     'foot_model': -1,
    #     'eye_model':  -1,
    #     'tail_model': -1,
    #     'tail_color': -1,
    #     'wing_model': -1
    # }

    # # -----------------------------------------------------------------
    # # TEST 0: SIŁA 100%
    # # -----------------------------------------------------------------
    # hierarchical_100_config = {
    #     0: { # Grupa kontrolna: Zawsze 0% siły, skrzydła i dziób losują się wolno
    #         'strength': 0.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {}
    #     },
    #     1: { # Grupa badawcza: 100% siły
    #         'strength': 1.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 2, 1: 1}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna 100%)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_100_config, 
    #     frozen_config=frozen_config_hier, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b100"
    # )

    #  # -----------------------------------------------------------------
    # # TEST 1: SIŁA 80%
    # # -----------------------------------------------------------------
    # hierarchical_80_config = {
    #     0: { 
    #         'strength': 0.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {}
    #     },
    #     1: { 
    #         'strength': 0.8, # Zmieniamy tylko to!
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 2, 1: 1}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna 60%)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_80_config, 
    #     frozen_config=frozen_config_hier, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b80"
    # )

    # # -----------------------------------------------------------------
    # # TEST 2: SIŁA 60%
    # # -----------------------------------------------------------------
    # hierarchical_60_config = {
    #     0: { 
    #         'strength': 0.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {}
    #     },
    #     1: { 
    #         'strength': 0.6, # Zmieniamy tylko to!
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 2, 1: 1}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna 60%)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_60_config, 
    #     frozen_config=frozen_config_hier, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b60"
    # )

    # # -----------------------------------------------------------------
    # # TEST 3: SIŁA 40%
    # # -----------------------------------------------------------------
    # hierarchical_40_config = {
    #     0: { 
    #         'strength': 0.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {}
    #     },
    #     1: { 
    #         'strength': 0.4, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 2, 1: 1}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna 40%)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_40_config, 
    #     frozen_config=frozen_config_hier, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b40"
    # )

    # # -----------------------------------------------------------------
    # # TEST 4: SIŁA 20%
    # # -----------------------------------------------------------------
    # hierarchical_20_config = {
    #     0: { 
    #         'strength': 0.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {}
    #     },
    #     1: { 
    #         'strength': 0.2, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 2, 1: 1}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna 20%)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_20_config, 
    #     frozen_config=frozen_config_hier, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b20"
    # )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 1: Użycie prawdopodobieństw
    # # -----------------------------------------------------------------
    # prob_scenario_config = {
    #     0: { # Klasa 0
    #         # Dziób: 20% model 0 (beak01), 60% model 1 (beak02), 20% model 2 (beak03)
    #         'beak_model': {'options': [0, 1, 2], 'probs': [0.2, 0.6, 0.2]},
    #         # Nogi: 90% model 0, 10% model 1
    #         'foot_model': {'options': [0, 1], 'probs': [0.9, 0.1]}
    #     },
    #     1: { # Klasa 1
    #         'beak_model': {'options': [0, 3], 'probs': [0.5, 0.5]},
    #         'foot_model': {'options': [2, 3], 'probs': [0.2, 0.8]}
    #     }
    # }
    
    # print("Generowanie datasetu metodą 1 (Probabilistyczna)...")
    # generate_probabilistic_dataset(
    #     prob_config=prob_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_probabilistyczne"
    # )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 2: Użycie hierarchiczności / stopnia zależności
    # # -----------------------------------------------------------------
    # hierarchical_scenario_config = {
    #     0: { # Klasa 0
    #         'strength': 0.1, # 10% szans, że zależność w ogóle się aktywuje dla danego ptaka
    #         'primary_feature': 'beak_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 1 # JEŚLI wylosuje się beak01 (indeks 0) I aktywuje się siła (strength), to tail_color MUSI być tail02 (indeks 1)
    #         }
    #     },
    #     1: { # Klasa 1
    #         'strength': 0.8, # 80% szans na aktywację zależności
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {
    #             0: 2, # Czerwone skrzydła wymuszają czerwony dziób
    #             1: 1  # Zielone skrzydła wymuszają czarny dziób
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_hierarchiczne"
    # )


    # # -----------------------------------------------------------------
    # # SCENARIUSZ 2: Użycie hierarchiczności / stopnia zależności
    # # -----------------------------------------------------------------
    # hierarchical_scenario_config = {
    #     0: { # Klasa 0
    #         'strength': 0.1, # 10% szans, że zależność w ogóle się aktywuje dla danego ptaka
    #         'primary_feature': 'beak_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 1 # JEŚLI wylosuje się beak01 (indeks 0) I aktywuje się siła (strength), to tail_color MUSI być tail02 (indeks 1)
    #         }
    #     },
    #     1: { # Klasa 1
    #         'strength': 1.0, # 80% szans na aktywację zależności
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {
    #             0: 2, # Czerwone skrzydła wymuszają czerwony dziób
    #             1: 1  # Zielone skrzydła wymuszają czarny dziób
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b100"
    # )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 2: Użycie hierarchiczności / stopnia zależności
    # # -----------------------------------------------------------------
    # hierarchical_scenario_config = {
    #     0: { # Klasa 0
    #         'strength': 0.1, # 10% szans, że zależność w ogóle się aktywuje dla danego ptaka
    #         'primary_feature': 'beak_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 1 # JEŚLI wylosuje się beak01 (indeks 0) I aktywuje się siła (strength), to tail_color MUSI być tail02 (indeks 1)
    #         }
    #     },
    #     1: { # Klasa 1
    #         'strength': 0.6, # 80% szans na aktywację zależności
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {
    #             0: 2, # Czerwone skrzydła wymuszają czerwony dziób
    #             1: 1  # Zielone skrzydła wymuszają czarny dziób
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b60"
    # )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 2: Użycie hierarchiczności / stopnia zależności
    # # -----------------------------------------------------------------
    # hierarchical_scenario_config = {
    #     0: { # Klasa 0
    #         'strength': 0.1, # 10% szans, że zależność w ogóle się aktywuje dla danego ptaka
    #         'primary_feature': 'beak_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 1 # JEŚLI wylosuje się beak01 (indeks 0) I aktywuje się siła (strength), to tail_color MUSI być tail02 (indeks 1)
    #         }
    #     },
    #     1: { # Klasa 1
    #         'strength': 0.4, # 80% szans na aktywację zależności
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {
    #             0: 2, # Czerwone skrzydła wymuszają czerwony dziób
    #             1: 1  # Zielone skrzydła wymuszają czarny dziób
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b40"
    # )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 2: Użycie hierarchiczności / stopnia zależności
    # # -----------------------------------------------------------------
    # hierarchical_scenario_config = {
    #     0: { # Klasa 0
    #         'strength': 0.1, # 10% szans, że zależność w ogóle się aktywuje dla danego ptaka
    #         'primary_feature': 'beak_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 1 # JEŚLI wylosuje się beak01 (indeks 0) I aktywuje się siła (strength), to tail_color MUSI być tail02 (indeks 1)
    #         }
    #     },
    #     1: { # Klasa 1
    #         'strength': 0.2, # 80% szans na aktywację zależności
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {
    #             0: 2, # Czerwone skrzydła wymuszają czerwony dziób
    #             1: 1  # Zielone skrzydła wymuszają czarny dziób
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu metodą 2 (Hierarchiczna)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_scenario_config, 
    #     frozen_config=frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_5b20"
    # )


    frozen_config_bez_szumu = {
        'beak_model': -1,
        'beak_color': -1,
        'tail_model': -1,
        'tail_color': -1,
        'wing_model': -1,
        'wing_color': -1
    }    
      # -----------------------------------------------------------------
    # SCENARIUSZ 3: Probabilistyczny bez szumu
    # klasa 0 i 1 różnią się tylko preferencjami co do oczu i nóg, a reszta cech zmienia sie losowo
    # -----------------------------------------------------------------
    prob_complex_config = {
        0: { # Klasa 0
            # Preferuje oczy 01 (80%), rzadko 02 (20%), nigdy 03
            'eye_model': {'options': [0, 1], 'probs': [0.8, 0.2]},
            # Zawsze ma specyficzny zestaw nóg (np. fioletowe/krótkie - indeksy 0 i 1)
            'foot_model': {'options': [0, 1], 'probs': [0.5, 0.5]}
        },
        1: { # Klasa 1
            # Preferuje oczy 03 (80%), rzadko 02 (20%), nigdy 01
            'eye_model': {'options': [2, 1], 'probs': [0.8, 0.2]},
            # Zawsze ma inny zestaw nóg (indeksy 2 i 3)
            'foot_model': {'options': [2, 3], 'probs': [0.5, 0.5]}
        }
    }
    
    print("Generowanie datasetu 3 (Probabilistyczny - dużo zmiennych)...")
    generate_probabilistic_dataset(
        prob_config=prob_complex_config, 
        frozen_config=frozen_config_bez_szumu, 
        num_classes=NUM_CLASSES, 
        samples_per_class=SAMPLES_PER_CLASS, 
        root_path="./testy_4b"
    )

    empty_frozen_config = {}


    # # -----------------------------------------------------------------
    # # SCENARIUSZ 3: Probabilistyczny bez mrozenia cech
    # # klasa 0 i 1 różnią się tylko preferencjami co do oczu i nóg, a reszta cech zmienia sie losowo
    # # -----------------------------------------------------------------
    prob_complex_config = {
        0: { # Klasa 0
            # Preferuje oczy 01 (80%), rzadko 02 (20%), nigdy 03
            'eye_model': {'options': [0, 1], 'probs': [1.0, 0.0]},
            # Zawsze ma specyficzny zestaw nóg (np. fioletowe/krótkie - indeksy 0 i 1)
            'foot_model': {'options': [0, 1], 'probs': [0.5, 0.5]}
        },
        1: { # Klasa 1
            # Preferuje oczy 03 (80%), rzadko 02 (20%), nigdy 01
            'eye_model': {'options': [2, 1], 'probs': [1.0, 0.0]},
            # Zawsze ma inny zestaw nóg (indeksy 2 i 3)
            'foot_model': {'options': [2, 3], 'probs': [0.5, 0.5]}
        }
    }
    
    print("Generowanie datasetu 3 (Probabilistyczny - dużo zmiennych)...")
    generate_probabilistic_dataset(
        prob_config=prob_complex_config, 
        frozen_config=empty_frozen_config, 
        num_classes=NUM_CLASSES, 
        samples_per_class=SAMPLES_PER_CLASS, 
        root_path="./testy_prob100_szum"
    )

    # # -----------------------------------------------------------------
    # # SCENARIUSZ 4: Hierarchiczny bez mrozenia cech
    # # -----------------------------------------------------------------
    # hierarchical_complex_config = {
    #     0: { # Klasa 0
    #         'strength': 0.0, 
    #         'primary_feature': 'tail_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {}
    #     },
    #     1: { # Klasa 1
    #         # 100% pewności, że jeśli ptak wylosuje konkretny ogon, będzie miał konkretny kolor.
    #         'strength': 1.0, 
    #         'primary_feature': 'tail_model',
    #         'dependent_feature': 'tail_color',
    #         'forced_combo': {
    #             0: 0, # Ogon 01 ZAWSZE jest czerwony (0)
    #             1: 1, # Ogon 02 ZAWSZE jest zielony (1)
    #             2: 2  # Ogon 03 ZAWSZE jest niebieski (2)
    #         }
    #     }
    # }
    
    # print("Generowanie datasetu 4 (Hierarchiczny - ukryta zasada)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_complex_config, 
    #     frozen_config=empty_frozen_config, 
    #     num_classes=NUM_CLASSES, 
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_hier_szum"
    # )

    # # =================================================================
    # # SCENARIUSZ: HIERARCHIA BASIC NA KOLORACH (3 KLASY)
    # # =================================================================
    
    # # Mrozimy kształty i inne kolory, żeby model skupił się tylko na badanej relacji
    # frozen_config_kolory = {
    #     'beak_model': -1,
    #     'foot_model': -1,
    #     'eye_model':  -1,
    #     'tail_model': -1,
    #     'tail_color': -1,
    #     'wing_model': -1
    # }

    # hierarchical_basic_colors_config = {
    #     0: { # Klasa 0: Dziób 0 wtedy zawsze nogi 0 (Tłumaczenie: Skrzydła 0 -> zawsze Dziób 0)
    #         'strength': 1.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {0: 0} # Czerwone skrzydła wymuszają żółty dziób
    #     },
    #     1: { # Klasa 1: Dziób 1 wtedy zawsze nogi 0 (Tłumaczenie: Skrzydła 1 -> zawsze Dziób 0)
    #         'strength': 1.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {1: 0} # Zielone skrzydła wymuszają żółty dziób
    #     },
    #     2: { # Klasa 2: Dziób 1 wtedy zawsze nogi 1 (Tłumaczenie: Skrzydła 1 -> zawsze Dziób 1)
    #         'strength': 1.0, 
    #         'primary_feature': 'wing_color',
    #         'dependent_feature': 'beak_color',
    #         'forced_combo': {1: 1} # Zielone skrzydła wymuszają czarny dziób
    #     }
    # }
    
    # print("Generowanie datasetu: Hierarchia Basic na kolorach (3 klasy)...")
    # generate_hierarchical_dataset(
    #     hierarchy_rules=hierarchical_basic_colors_config, 
    #     frozen_config=frozen_config_kolory, 
    #     num_classes=3, # <-- WAŻNE: Wymuszamy 3 klasy dla tego konkretnego zbioru!
    #     samples_per_class=SAMPLES_PER_CLASS, 
    #     root_path="./testy_hierarchia_kolory"
    # )