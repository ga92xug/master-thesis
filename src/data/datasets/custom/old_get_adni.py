
def get_adni(
    data_dir: str,
    name: str,
    **kwargs
) -> Tuple[Dict, Dict]:
    def get_subject_identifier(filename: str) -> str:
        return filename.split('_')[1]
    
    location = path.join(data_dir, "adni/adni1/slice2Ddata")
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []
    for i, class_name in enumerate(os.listdir(location)):
        class_path = path.join(location, class_name)
        if not path.isdir(class_path):
            continue
        class_images = [f for f in os.listdir(class_path) if f.startswith('axial')]
        subject_images = {}
        for image in class_images:
            subject_id = get_subject_identifier(image)
            if subject_id not in subject_images:
                subject_images[subject_id] = []
            subject_images[subject_id].append(image)
        subjects = list(subject_images.keys())

        train_subjects, test_subjects = train_test_split(subjects, test_size=0.7, random_state=42)
        val_subjects, test_subjects = train_test_split(test_subjects, test_size=0.5, random_state=42)

        for subject in train_subjects:
            train_images.extend([path.join(class_path, img) for img in subject_images[subject]])
            train_labels.extend([i] * len(subject_images[subject]))
        
        for subject in val_subjects:
            val_images.extend([path.join(class_path, img) for img in subject_images[subject]])
            val_labels.extend([i] * len(subject_images[subject]))
        
        for subject in test_subjects:
            test_images.extend([path.join(class_path, img) for img in subject_images[subject]])
            test_labels.extend([i] * len(subject_images[subject]))


    images = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    #image = np.load(train_images[0])
    # check if values are in the range [0, 1]
    #print("max val", np.max(image), "min val", np.min(image))
    #quit()
    
    labels = {
        "train": train_labels,
        "val": val_labels,
        "test": test_labels,
    }
    return images, labels


def get_mammo(
    data_dir: str,
    name: str,
    resolution: int,
) -> Tuple[Dict, Dict]:
    
    data_dir = path.join(data_dir, "mammo/")

    subject_functions = {
        "DDSM": lambda x: x.split('.')[0],
        "INbreast": lambda x: x.split(' ')[0],
        #"MIAS": lambda x: x.split('.')[0],
    }

    # options 
    # DDSM, INbreast, MIAS
    # train DDSM
    location = path.join(data_dir, "INbreast")
    print("INbreast")
    images, labels = subject_split_image_label_folder(
        location, 
        subject_func=subject_functions["INbreast"], 
        splits=[0.8],
    )

    train_images, train_labels = images[0], labels[0]
    val_images, val_labels = images[1], labels[1]

    # check that train and val are disjoint
    #assert len(set(train_images).intersection(set(val_images))) == 0

    # test INbreast
    print("DDSM")
    location = path.join(data_dir, "DDSM")
    test_images, test_labels = subject_split_image_label_folder(
        location,
        subject_func=subject_functions["DDSM"],
    )
    test_images, test_labels = test_images[0], test_labels[0]

    # check that test and train are disjoint
    #assert len(set(test_images).intersection(set(train_images))) == 0
    # check that test and val are disjoint
    #assert len(set(test_images).intersection(set(val_images))) == 0

    images = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    labels = {
        "train": train_labels,
        "val": val_labels,
        "test": test_labels,
    }

    return images, labels
