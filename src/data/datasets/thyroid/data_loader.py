"""
df_train=pd.read_csv("dataframe/thyroid_train_fold"+str(i+1)+".csv")
    df_val=pd.read_csv("dataframe/thyroid_val_fold"+str(i+1)+".csv")
    train_data_generator = ImageDataGenerator(
                                    rescale=1./255,
                                    preprocessing_function=preprocess_input,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        x_col = 'img_dir',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col = 'img_dir',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')   
    num_classes =len(train_generator.class_indices)
"""