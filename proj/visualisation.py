import matplotlib.pyplot as plt
import numpy as np

# Data from your CatBoost training

def iterations_graph():
    iterations = [1000, 1500, 2000, 2500, 3000]
    targets = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
    r2_values = [
        [0.545, 0.508, 0.647, 0.375, 0.423, 0.500],  # 1000 iterations
        [0.548, 0.514, 0.648, 0.375, 0.430, 0.504],  # 1500 iterations
        [0.552, 0.518, 0.649, 0.376, 0.432, 0.505],  # 2000 iterations
        [0.555, 0.522, 0.650, 0.374, 0.436, 0.507],  # 2500 iterations
        [0.556, 0.524, 0.651, 0.372, 0.436, 0.508],  # 3000 iterations
    ]
    mean_r2_values = [0.500, 0.503, 0.505, 0.507, 0.508]

    # Plot R² values for each target across iterations
    plt.figure(figsize=(10, 6))
    for i, target in enumerate(targets):
        plt.plot(iterations, [r2[i] for r2 in r2_values], label=target)

    plt.plot(iterations, mean_r2_values, 'k--', label='Mean R²', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('R²')
    plt.title('R² Values Across Different Iterations: CatBoost(ViT-L)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def XGBoost_CatBoost_graph():
    # Data from your results
    targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]
    r2_xgboost_vitb = [0.426, 0.418, 0.607, 0.277, 0.329, 0.451]
    r2_xgboost_vitl = [0.481, 0.469, 0.606, 0.365, 0.375, 0.475]
    r2_catboost_vitl = [0.548, 0.514, 0.648, 0.375, 0.430, 0.504]
    r2_catboost_vitb = [0.481, 0.479, 0.636, 0.341, 0.389, 0.482]

    mean_r2_xgboost_vitb = 0.418
    mean_r2_xgboost_vitl = 0.462
    mean_r2_catboost_vitl = 0.503
    mean_r2_catboost_vitb = 0.468

    # Set up the plot
    x = np.arange(len(targets))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, r2_xgboost_vitb, width, label='XGBoost vitb')
    rects2 = ax.bar(x, r2_xgboost_vitl, width, label='XGBoost vitl')
    rects3 = ax.bar(x + width, r2_catboost_vitb, width, label='CatBoost vitb')
    rects4 = ax.bar(x + 2 * width, r2_catboost_vitl, width, label='CatBoost vitl')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Targets')
    ax.set_ylabel('R²')
    ax.set_title('Comparison of R² values for XGBoost and CatBoost')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(targets)
    ax.legend()

    fig.tight_layout()

    # Show the plot
    plt.show()

    # Plot the mean R² values for a clearer comparison
    models = ['XGBoost vitb', 'XGBoost vitl', 'CatBoost vitb', 'CatBoost vitl']
    mean_r2_values = [mean_r2_xgboost_vitb, mean_r2_xgboost_vitl, mean_r2_catboost_vitb, mean_r2_catboost_vitl]

    plt.figure(figsize=(8, 5))
    plt.bar(models, mean_r2_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    plt.xlabel('Models')
    plt.ylabel('Mean R²')
    plt.title('Mean R² Comparison for XGBoost and CatBoost')
    plt.show()

def feature_extractor_size_graph():
    # Data for R² values
    targets = ["X4", "X11", "X18", "X26", "X50", "X3112"]
    r2_vits = [0.415, 0.425, 0.580, 0.298, 0.319, 0.443]
    r2_vitb = [0.481, 0.479, 0.636, 0.341, 0.389, 0.482]
    r2_vitl = [0.548, 0.514, 0.648, 0.375, 0.430, 0.504]
    r2_vitg = [0.553, 0.531, 0.638, 0.354, 0.442, 0.514]

    mean_r2_vits = 0.413
    mean_r2_vitb = 0.468
    mean_r2_vitl = 0.503
    mean_r2_vitg = 0.505

    # Set up the plot
    x = np.arange(len(targets))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width*1.5, r2_vits, width, label='vits')
    rects2 = ax.bar(x - width/2, r2_vitb, width, label='vitb')
    rects3 = ax.bar(x + width/2, r2_vitl, width, label='vitl')
    rects4 = ax.bar(x + width*1.5, r2_vitg, width, label='vitg')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Targets')
    ax.set_ylabel('R²')
    ax.set_title('Comparison of R² values for different vit sizes (1500 iterations)')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()

    fig.tight_layout()

    # Show the plot
    plt.show()

    # Plot the mean R² values for a clearer comparison
    models = ['vits', 'vitb', 'vitl', 'vitg']
    mean_r2_values = [mean_r2_vits, mean_r2_vitb, mean_r2_vitl, mean_r2_vitg]

    plt.figure(figsize=(8, 5))
    plt.bar(models, mean_r2_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
    plt.xlabel('Models')
    plt.ylabel('Mean R²')
    plt.title('Mean R² Comparison for different vit sizes (1500 iterations)')
    plt.show()

if __name__ == "__main__":
    feature_extractor_size_graph()
