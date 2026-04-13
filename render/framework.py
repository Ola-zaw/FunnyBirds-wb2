import json
import random
import math
import os

def build_experiment_config(num_classes, scenario):
    """
    Tłumaczy konfigurację (macierz) na język zrozumiały dla generatora JSON.
    -1 oznacza 'zamrożone' cechy. Inne liczby (0, 1, 2...) to indeksy części z bazy wiedzy.
    """
    
    knowledge_base = {
        'beak_model': ['beak01.glb', 'beak02.glb', 'beak03.glb', 'beak04.glb'], # 4 opcje (0, 1, 2, 3)
        'beak_color': ['yellow', 'black', 'red', 'green'], 
        'eye_model':  ['eye01.glb', 'eye02.glb', 'eye03.glb'],
        'foot_model': ['foot01.glb', 'foot02.glb', 'foot03.glb', 'foot04.glb'],
        'tail_model': ['tail01.glb', 'tail02.glb', 'tail03.glb', 'tail04.glb', 'tail05.glb', 'tail06.glb', 'tail07.glb', 'tail08.glb', 'tail09.glb'],
        'tail_color': ['red', 'green', 'blue'],
        'wing_model': ['wing01.glb', 'wing02.glb', 'wing03.glb', 'wing04.glb', 'wing05.glb', 'wing06.glb'],
        'wing_color': ['red', 'green', 'blue', 'yellow']
    }

    dependent_features = {i: {} for i in range(num_classes)}
    frozen_features = {}

    # Mapowanie opcji na konkretne ścieżki modeli 3D
    for feature, values in scenario.items():
        if values == [-1] or values[0] == -1:
            # Bierzemy domyślną opcję i zamrażamy
            frozen_features[feature] = knowledge_base[feature][0] 
        else:
            for class_id in range(num_classes):
                option_index = values[class_id]
                dependent_features[class_id][feature] = knowledge_base[feature][option_index]

    return dependent_features, frozen_features


def generate_json_dataset(dependent_features, frozen_features, samples_per_class, root_path="./my_experiment", mode="train"):
    """
    Tworzy końcowy plik JSON i zapisuje go bezpośrednio w wymaganym folderze FunnyBirds.
    """
    dataset = []
    
    # 1. Budowanie struktury ptaków
    for class_id, class_features in dependent_features.items():
        for _ in range(samples_per_class):
            sample = {'class_idx': class_id}
            
            # Cechy zamrożone i klasowe
            for key, val in frozen_features.items():
                sample[key] = val
            for key, val in class_features.items():
                sample[key] = val

            # Kamery i tło
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
            
            dataset.append(sample)

    # 2. Tworzenie folderów
    target_dir = os.path.join(root_path, "FunnyBirds")
    os.makedirs(target_dir, exist_ok=True) 
    
    file_name = f"dataset_{mode}.json"
    full_path = os.path.join(target_dir, file_name)

    with open(full_path, "w") as outfile:
        json.dump(dataset, outfile)
    print(f"Zapisano {len(dataset)} ptaków bezpośrednio do folderu: {full_path}")


# =====================================================================
# Scenariusze
# =====================================================================

NUM_CLASSES = 3
SAMPLES_PER_CLASS = 25
EXPERIMENT_FOLDER = "./testy_hier"

# SCENARIUSZ 0: A (0), B (1) i C (2)
# scenario_config = {
#     # Klasy ID:    [ 0,  1,  2 ]
#     'beak_model':  [ 0,  1,  2 ], # Klasa 0->Dziób A, Klasa 1->Dziób B, Klasa 2->Dziób C
#     'foot_model':  [ 0,  1,  2 ], # Nogi A, Nogi B, Nogi C
#     'wing_model':  [ 0,  2,  0 ], # Opcja A dla Klasy 0 i 2, Opcja C dla Klasy 1
#     'beak_color':  [-1],          # Reszta zamrożona na sztywno
#     'eye_model':   [-1], 
#     'tail_model':  [-1], 
#     'tail_color':  [-1], 
#     'wing_color':  [-1]
# }

# SCENARIUSZ 1: NIEZALEŻNOŚĆ
# Klasa 0 (AA): Dziób 0, Nogi 0
# Klasa 1 (AB): Dziób 0, Nogi 1
# Klasa 2 (BA): Dziób 1, Nogi 0
# Klasa 3 (BB): Dziób 1, Nogi 1
# scenario_config = {
#     'beak_model': [0, 0, 1, 1], 
#     'foot_model': [0, 1, 0, 1], 
#     'beak_color': [-1],
#     'eye_model':  [-1], 
#     'tail_model': [-1], 
#     'tail_color': [-1], 
#     'wing_model': [-1], 
#     'wing_color': [-1]
# }

# SCENARIUSZ 2: 100% WYNIKANIE (Dziób determinuje nogi, 2 klasy)
# Klasa 0 (AA): Dziób 0 zawsze z Nogami 0
# Klasa 1 (BB): Dziób 1 zawsze z Nogami 1
# scenario_config = {
#     'beak_model': [0, 1], 
#     'foot_model': [0, 1], 
#     'beak_color': [-1], 
#     'eye_model':  [-1], 
#     'tail_model': [-1], 
#     'tail_color': [-1], 
#     'wing_model': [-1], 
#     'wing_color': [-1]
# }

# SCENARIUSZ 3: JEDNOKIERUNKOWA ZALEŻNOŚĆ A->A, z B cokolwiek
# Klasa 0 (AA): Dziób 0, Nogi 0
# Klasa 1 (BA): Dziób 1, Nogi 0
# Klasa 2 (BB): Dziób 1, Nogi 1
scenario_config = {
    'beak_model': [0, 1, 1], 
    'foot_model': [0, 0, 1], 
    'beak_color': [-1], 
    'eye_model':  [-1], 
    'tail_model': [-1], 
    'tail_color': [-1], 
    'wing_model': [-1], 
    'wing_color': [-1]
}

dep_features, froz_features = build_experiment_config(NUM_CLASSES, scenario_config)
generate_json_dataset(dep_features, froz_features, SAMPLES_PER_CLASS, root_path=EXPERIMENT_FOLDER)