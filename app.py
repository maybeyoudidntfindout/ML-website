import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)

st.set_page_config(page_title="ML Model Explorer", layout="centered")

st.title("ML Model Explorer")
st.write(
    "Explore the methods from our notebooks. Test-set results show how a "
    "model performed; sliders let you try new examples."
)

section = st.radio(
    "Choose a model",
    ["Linear regression", "Perceptron"],
    horizontal=True,
)


def regression_scores(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R²": r2_score(actual, predicted),
    }


def train_scratch(X_train, y_train, learning_rate, epochs):
    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    errors_by_epoch = []

    for _ in range(epochs):
        errors = 0
        for row, label in zip(X_train, y_train):
            prediction = int(np.dot(row, weights) + bias >= 0)
            error = label - prediction
            weights += learning_rate * error * row
            bias += learning_rate * error
            errors += int(error != 0)
        errors_by_epoch.append(errors)

    return weights, bias, errors_by_epoch


def scratch_predict(samples, weights, bias):
    return np.array([
        int(np.dot(row, weights) + bias >= 0)
        for row in samples
    ])


# ---------------- LINEAR REGRESSION ----------------

if section == "Linear regression":
    st.header("One feature or several?")
    st.write(
        "**SLR** uses one feature to predict a number. **MLR** uses several. "
        "The comparison uses the same dataset and test split for both."
    )

    dataset_choice = st.selectbox("Dataset", ["Diabetes", "Iris"])

    if dataset_choice == "Diabetes":
        dataset = load_diabetes()
        X = dataset.data
        y = dataset.target
        feature_names = list(dataset.feature_names)
        target_name = "progression score"
        simple_default = "bmi"
        multiple_default = feature_names
        split_seed = 42
        st.info(
            "The Diabetes dataset contains transformed measurements. "
            "Its target is a progression score—not a diagnosis."
        )
    else:
        dataset = load_iris()
        X = dataset.data
        y = dataset.data[:, 0]
        feature_names = list(dataset.feature_names)
        target_name = "sepal length (cm)"
        simple_default = "petal length (cm)"
        multiple_default = [
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        split_seed = 42

    feature_index = {name: i for i, name in enumerate(feature_names)}

    st.subheader("Choose model inputs")
    simple_name = st.selectbox(
        "SLR uses this one feature",
        feature_names,
        index=feature_names.index(simple_default),
    )
    multiple_names = st.multiselect(
        "MLR uses these features together",
        feature_names,
        default=multiple_default,
    )

    if not multiple_names:
        st.warning("Choose at least one feature for MLR.")
        st.stop()

    simple_idx = [feature_index[simple_name]]
    multi_idx = [feature_index[name] for name in multiple_names]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X[:, simple_idx], y, test_size=0.2, random_state=split_seed
    )
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X[:, multi_idx], y, test_size=0.2, random_state=split_seed
    )

    slr = LinearRegression().fit(Xs_train, ys_train)
    mlr = LinearRegression().fit(Xm_train, ym_train)
    ys_pred = slr.predict(Xs_test)
    ym_pred = mlr.predict(Xm_test)

    slr_scores = regression_scores(ys_test, ys_pred)
    mlr_scores = regression_scores(ym_test, ym_pred)

    st.subheader("Test-set comparison")
    st.caption(
        "These scores are fixed for the selected dataset and features. "
        "They don’t change when you adjust the example sliders."
    )
    score_table = pd.DataFrame(
        [slr_scores, mlr_scores],
        index=["SLR — one feature", "MLR — selected features"],
    )
    st.dataframe(score_table.round(3), use_container_width=True)
    st.caption("Lower MAE/RMSE is better. R² closer to 1 is generally better.")

    st.subheader("Try a new example")
    st.write(
        "Adjust the feature values to see the predictions change. "
        "The graphs mark your current example."
    )

    feature_values = X[:, simple_idx[0]]
    x_min = float(feature_values.min())
    x_max = float(feature_values.max())

    simple_value = st.slider(
        f"SLR input: {simple_name}",
        min_value=x_min,
        max_value=x_max,
        value=float(np.median(feature_values)),
    )
    simple_prediction = float(slr.predict([[simple_value]])[0])
    st.metric(f"SLR predicted {target_name}", f"{simple_prediction:.2f}")

    st.subheader("SLR: input feature and fitted line")
    st.caption(
        "The slider sets the x-value. Blue dots are actual test examples, "
        "the blue line shows the fitted relationship, and the orange diamond "
        "shows the prediction for your slider value."
    )

    line_x = np.linspace(x_min, x_max, 100)
    line_y = slr.predict(line_x.reshape(-1, 1))

    slr_fig = go.Figure()
    slr_fig.add_trace(go.Scatter(
        x=Xs_test[:, 0],
        y=ys_test,
        mode="markers",
        name="Actual test examples",
        marker=dict(color="#94a3b8", size=8, opacity=0.7),
    ))
    slr_fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="SLR fitted line",
        line=dict(color="#2563eb", width=3),
    ))
    slr_fig.add_trace(go.Scatter(
        x=[simple_value],
        y=[simple_prediction],
        mode="markers",
        name="Your slider prediction",
        marker=dict(color="#f97316", size=14, symbol="diamond"),
    ))
    slr_fig.update_layout(
        xaxis_title=simple_name,
        yaxis_title=target_name,
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(slr_fig, use_container_width=True)

    st.subheader("MLR: try several inputs")
    st.write(
        "Each slider sets one feature. Changing any of them updates the "
        "MLR prediction."
    )

    user_values = []
    for idx in multi_idx:
        values = X[:, idx]
        user_values.append(
            st.slider(
                f"MLR input: {feature_names[idx]}",
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(np.median(values)),
                key=f"mlr-{dataset_choice}-{feature_names[idx]}",
            )
        )

    multi_prediction = float(mlr.predict([user_values])[0])
    st.metric(f"MLR predicted {target_name}", f"{multi_prediction:.2f}")

    st.subheader("MLR: input feature and prediction slice")
    st.caption(
        "The line shows predictions as one feature changes while the other "
        "features stay at your slider settings. Gray dots are actual test "
        "examples; their other feature values vary, so they may not lie on "
        "this particular line."
    )

    effect_feature = st.selectbox(
        "Feature to vary in the graph",
        multiple_names,
        key="mlr_graph_feature",
    )
    effect_position = multiple_names.index(effect_feature)
    effect_index = multi_idx[effect_position]
    feature_min = float(X[:, effect_index].min())
    feature_max = float(X[:, effect_index].max())

    sweep_values = np.linspace(feature_min, feature_max, 100)
    sweep_inputs = np.tile(np.array(user_values), (100, 1))
    sweep_inputs[:, effect_position] = sweep_values
    sweep_predictions = mlr.predict(sweep_inputs)

    # Actual test examples: selected feature value vs. actual target value.
    actual_x = Xm_test[:, effect_position]

    mlr_fig = go.Figure()
    mlr_fig.add_trace(go.Scatter(
        x=actual_x,
        y=ym_test,
        mode="markers",
        name="Actual test examples",
        marker=dict(color="#94a3b8", size=8, opacity=0.65),
    ))
    mlr_fig.add_trace(go.Scatter(
        x=sweep_values,
        y=sweep_predictions,
        mode="lines",
        name="MLR prediction slice",
        line=dict(color="#0f766e", width=3),
    ))
    mlr_fig.add_trace(go.Scatter(
        x=[user_values[effect_position]],
        y=[multi_prediction],
        mode="markers",
        name="Your current inputs",
        marker=dict(color="#f97316", size=14, symbol="diamond"),
    ))
    mlr_fig.update_layout(
        xaxis_title=effect_feature,
        yaxis_title=target_name,
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(mlr_fig, use_container_width=True)


# ---------------- PERCEPTRON CLASSIFICATION ----------------

else:
    st.header("Iris classification with Perceptrons")
    st.write(
        "These exercises use petal length and petal width. The notebook "
        "compares binary classification with three-class classification, "
        "and includes a scikit-learn Perceptron and versions from scratch."
    )

    task = st.radio(
        "Classification task",
        ["Setosa vs. Versicolor", "All three Iris classes"],
        horizontal=True,
    )
    method = st.selectbox(
        "Model implementation",
        ["Scikit-learn", "From scratch"],
    )

    if method == "From scratch":
        learning_rate = st.slider(
            "Learning rate",
            0.01, 1.0, 0.1, 0.01,
            help="Controls the size of each weight update during training.",
        )
        epochs = st.slider(
            "Training epochs",
            10, 300, 100, 10,
            help="One epoch is one pass through the training examples.",
        )
    else:
        st.caption(
            "The scikit-learn model uses the settings from your notebook. "
            "Learning-rate and epoch controls apply only to the scratch model."
        )

    iris = load_iris()
    X_all = iris.data[:, 2:4]
    y_all = iris.target

    if task == "Setosa vs. Versicolor":
        keep = y_all < 2
        X, y = X_all[keep], y_all[keep]
        class_names = list(iris.target_names[:2])
    else:
        X, y = X_all, y_all
        class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    error_history = None

    if method == "Scikit-learn":
        model = Perceptron(eta0=0.1, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        def predict_samples(samples):
            return model.predict(samples)

    elif task == "Setosa vs. Versicolor":
        weights, bias, error_history = train_scratch(
            X_train, y_train, learning_rate, epochs
        )

        def predict_samples(samples):
            return scratch_predict(samples, weights, bias)

    else:
        models = {}
        error_history = {}

        for class_id in range(3):
            binary_labels = (y_train == class_id).astype(int)
            weights, bias, errors = train_scratch(
                X_train, binary_labels, learning_rate, epochs
            )
            models[class_id] = (weights, bias)
            error_history[class_id] = errors

        def predict_samples(samples):
            predicted = []
            for row in samples:
                scores = [
                    np.dot(row, models[c][0]) + models[c][1]
                    for c in range(3)
                ]
                predicted.append(int(np.argmax(scores)))
            return np.array(predicted)

    test_predictions = predict_samples(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    st.metric("Test-set accuracy", f"{accuracy:.1%}")
    st.caption(
        "This score uses held-out test examples, not the example you set below."
    )

    st.subheader("Confusion matrix")
    st.write(
        "Rows are actual classes; columns are predicted classes. "
        "The diagonal cells are correct predictions."
    )
    cm = confusion_matrix(y_test, test_predictions, labels=range(len(class_names)))
    matrix_fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale=["#eff6ff", "#2563eb"],
        labels=dict(
            x="Predicted class",
            y="Actual class",
            color="Examples",
        ),
    )
    matrix_fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(matrix_fig, use_container_width=True)

    st.subheader("Try a flower")
    st.write(
        "Change petal length or width. The predicted class and highlighted "
        "point on the map will update."
    )
    col1, col2 = st.columns(2)
    with col1:
        petal_length = st.slider("Petal length (cm)", 0.0, 7.0, 2.5, 0.1)
    with col2:
        petal_width = st.slider("Petal width (cm)", 0.0, 3.0, 0.8, 0.1)

    user_sample = np.array([[petal_length, petal_width]])
    user_class = int(predict_samples(user_sample)[0])
    st.success(f"Predicted class: **{class_names[user_class]}**")

    st.subheader("Decision map")
    st.caption(
        "Background color shows predicted classes across petal measurements. "
        "The X marks your selected flower."
    )

    x_min, x_max = float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5)
    y_min, y_max = float(X[:, 1].min() - 0.3), float(X[:, 1].max() + 0.3)
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 80)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.c_[xx.ravel(), yy.ravel()]
    regions = predict_samples(grid).reshape(xx.shape)

    decision_fig = go.Figure()
    decision_fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=regions,
        colorscale=["#bfdbfe", "#bbf7d0", "#fecaca"],
        opacity=0.45,
        showscale=False,
        hoverinfo="skip",
        contours=dict(start=0, end=max(1, len(class_names) - 1), size=1),
    ))

    for class_id, class_name in enumerate(class_names):
        decision_fig.add_trace(go.Scatter(
            x=X[y == class_id, 0],
            y=X[y == class_id, 1],
            mode="markers",
            name=class_name,
            marker=dict(size=7, opacity=0.65),
        ))

    decision_fig.add_trace(go.Scatter(
        x=[petal_length],
        y=[petal_width],
        mode="markers",
        name="Your flower",
        marker=dict(size=15, color="#111827", symbol="x"),
    ))
    decision_fig.update_layout(
        xaxis_title="Petal length (cm)",
        yaxis_title="Petal width (cm)",
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(decision_fig, use_container_width=True)

    if error_history is not None:
        st.subheader("Scratch model: training errors")
        error_fig = go.Figure()

        if task == "Setosa vs. Versicolor":
            error_fig.add_trace(go.Scatter(
                x=list(range(1, epochs + 1)),
                y=error_history,
                mode="lines",
                name="Training errors",
                line=dict(color="#7c3aed", width=3),
            ))
        else:
            for class_id, errors in error_history.items():
                error_fig.add_trace(go.Scatter(
                    x=list(range(1, epochs + 1)),
                    y=errors,
                    mode="lines",
                    name=f"{class_names[class_id]} vs. rest",
                ))

        error_fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Training errors",
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(error_fig, use_container_width=True)
