from sklearn.model_selection import train_test_split

x_train, x_temp, y_train, y_temp = train_test_split(faces, ids, train_size=0.7, random_state=0, stratify=ids)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.5,
    random_state=0,
    stratify=y_temp,
)
